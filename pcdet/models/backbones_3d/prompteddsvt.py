import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from math import ceil

from pcdet.models.model_utils.dsvt_utils import get_window_coors, get_inner_win_inds_cuda, get_pooling_index, get_continous_inds
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned

from .dsvt_addvoxel import DSVT, SetAttention, DSVT_EncoderLayer, _get_activation_fn


class PromptedDSVT(DSVT):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.
    '''

    def __init__(self, model_cfg, **kwargs):
        super(PromptedDSVT, self).__init__(model_cfg, **kwargs)

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''
        voxel_info = self.input_layer(batch_dict)

        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s+1}'] for s in range(self.stage_num-1)]

        output = voxel_feat
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}') # stage_0 = nn.ModuleList(block_list[DSVTBlock, DSVTBlock, DSVTBlock, DSVTBlock])
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}') # residual_norm_stage_0 = nn.ModuleList(norm_list[LayerNorm(192), LayerNorm(192), LayerNorm(192), LayerNorm(192)])
            for i in range(len(block_layers)): # block_layers
                block = block_layers[i]
                residual = output.clone()
                if self.use_torch_ckpt==False:
                    output = block(output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], \
                                block_id=block_id)
                else:
                    output = checkpoint(block, output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], block_id)
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats[stage_id].type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id], pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)
                
                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict


class DSVTBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()

        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
    ):
        num_shifts = 2
        output = src
        # TODO: bug to be fixed, mismatch of pos_embed
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed) # -> DSVT_EncoderLayer(src, set_voxel_inds, set_voxel_masks, pos=None)

        return output


class PromptedDSVT_EncoderLayer(DSVT_EncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0):
        super(PromptedDSVT_EncoderLayer, self).__init__(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0)
        
        self.win_attn = PromptedSetAttention(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self,src,set_voxel_inds,set_voxel_masks,pos=None):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds) # -> SetAttention(src, pos=None, key_padding_mask=None, voxel_inds=None)
        src = src + identity
        src = self.norm(src)

        return src


class PromptedSetAttention(SetAttention):
    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0):
        super(PromptedSetAttention, self).__init__(d_model, nhead, dropout, dim_feedforward=2048, 
                                                   activation="relu", batch_first=True, mlp_dropout=0)
        
        # prompt
        self.num_prompt = 50
        self.set_size = 36
        self.dim = 192
        self.prompt = nn.Parameter(torch.zeros(self.num_prompt, self.set_size, self.dim))

        self.activation = _get_activation_fn(activation)

    def incorporate_prompt(self, query, key, value):
        prompt_query = torch.cat((self.prompt, query), dim=0)
        prompt_key = torch.cat((self.prompt, key), dim=0)
        prompt_value = torch.cat((self.prompt, value), dim=0)
        return prompt_query, prompt_key, prompt_value

    def incorporate_mask(self, key_padding_mask):
        prompt_mask = torch.ones(self.num_prompt, self.set_size, dtype=torch.bool, device=key_padding_mask.device)
        prompt_padding_mask = torch.cat((prompt_mask, key_padding_mask), dim=0)
        return prompt_padding_mask
    
    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None):
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size). (?, 36)
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
        Returns:
            src (Tensor[float]): Voxel features.
        '''
        set_features = src[voxel_inds] # set_features.size: (set_num, set_size=36, C=192)
        if pos is not None:
            set_pos = pos[voxel_inds] # set_pos: (set_num, set_size=36, C=192)
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features

        # prompt
        prompt_query, prompt_key, prompt_value = self.incorporate_prompt(query, key, value)
        prompt_padding_mask = self.incorporate_mask(key_padding_mask)

        if key_padding_mask is not None:
            src2 = self.self_attn(prompt_query, prompt_key, prompt_value, prompt_padding_mask)[0] # -> nn.MultiheadAttention(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                                                                                                    #need_weights: bool = True, attn_mask: Optional[Tensor] = None)
        else:
            src2 = self.self_attn(prompt_query, prompt_key, prompt_value)[0]

        # map voxel featurs from set space to voxel space: (set_num, set_size, C) --> (N, C)
        flatten_inds = voxel_inds.reshape(-1)
        unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
        src2 = src2.reshape(-1, self.d_model)[perm]

        print('PromptedSetAttention')

        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    

# class promptedSetAttention(SetAttention):
#     def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0):
#         super().__init__(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        
#         # (set_num, set_size=36, C=192)
#         self.num_prompt = 50
#         self.set_size = 36
#         self.dim = 192
#         self.prompt = nn.Parameter(torch.zeros(self.num_prompt, self.set_size, self.dim))

#     def incorporate_prompt(self, query, key, value):
#         prompt_query = torch.cat((self.prompt, query), dim=0)
#         prompt_key = torch.cat((self.prompt, key), dim=0)
#         prompt_value = torch.cat((self.prompt, value), dim=0)
#         return prompt_query, prompt_key, prompt_value

#     def incorporate_mask(self, key_padding_mask):
#         prompt_mask = torch.ones(self.num_prompt, self.set_size, dtype=torch.bool)
#         prompt_padding_mask = torch.cat((prompt_mask, key_padding_mask), dim=0)
#         return prompt_padding_mask

#     def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None):
#         '''
#         Args:
#             src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
#             pos (Tensor[float]): Position embedding vectors with shape (N, C).
#             key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
#             voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
#         Returns:
#             src (Tensor[float]): Voxel features.
#         '''
#         set_features = src[voxel_inds] # set_features.size: (set_num, set_size=36, C=192)
#         if pos is not None:
#             set_pos = pos[voxel_inds] # set_pos: (set_num, set_size=36, C=192)
#         else:
#             set_pos = None
#         if pos is not None:
#             query = set_features + set_pos
#             key = set_features + set_pos
#             value = set_features

#         # Adding prompt
#         prompt_query, prompt_key, prompt_value = self.incorporate_prompt(query, key, value)
#         prompt_padding_mask = self.incorporate_mask(key_padding_mask)


#         if key_padding_mask is not None:
#             src2 = self.self_attn(prompt_query, prompt_key, prompt_value, prompt_padding_mask)[0] # -> nn.MultiheadAttention(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                                                                                                     #need_weights: bool = True, attn_mask: Optional[Tensor] = None)
#         else:
#             src2 = self.self_attn(prompt_query, prompt_key, prompt_value)[0]

#         # map voxel featurs from set space to voxel space: (set_num, set_size, C) --> (N, C)
#         flatten_inds = voxel_inds.reshape(-1)
#         unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
#         perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
#         inverse, perm = inverse.flip([0]), perm.flip([0])
#         perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
#         src2 = src2.reshape(-1, self.d_model)[perm]

#         print(f'src_prompt shape: ', src2.size())

#         # FFN layer
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)

#         return src