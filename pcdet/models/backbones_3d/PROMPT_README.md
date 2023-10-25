1. dsvt_addset:
    Adding extra sets in each windows.

2. dsvt_addvoxel:
    Adding extra voxels which concatenate on batch_dict['voxel_features']

3. dsvt_pg:
    Adding extra voxels like dsvt_voxel.
    Using a 4-layers prompt generator to generate a pi(1, 192).
    pi do expend and add on extra voxels.

4. dsvt_pg2:
    Directly Using a 4-layers prompt generator to generate a prompt(1, 192).
    concatenate prompt to each sets.

5. dsvt_addtoken:
    Adding extra tokens in each sets.