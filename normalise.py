# %%
#Load nifti file, transfomr to numpy, apply the normaliastion and save in different path


import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# %%
import numpy as np

def normalize_3d_voxel(input_data, range=(-1, 1), percentiles=(1, 99)):
    """
    Normalize a 3D voxel with percentiles.

    :param input_data: 3D numpy array to be normalized.
    :param range: min and max output values.
    :param percentiles: lower and upper percentile to clip the data.
    :return: Normalized 3D numpy array.
    """
    
    # Ensure input_data is a numpy array
    if not isinstance(input_data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    
    # Get the cutoff values for the entire 3D voxel
    cutoff = np.percentile(input_data, percentiles)
    
    # Clip the voxel values based on the cutoff
    np.clip(input_data, *cutoff, out=input_data)
    
    # Normalize the entire 3D voxel
    in_min = input_data.min()
    in_max = input_data.max()
    out_min, out_max = range
    
    input_data -= in_min
    if in_max - in_min == 0:
        # handle this case accordingly; e.g., set all values to out_min
        input_data[:] = out_min
    else:
        input_data /= (in_max - in_min)
        input_data *= (out_max - out_min)
        input_data += out_min

    return input_data


#folder path:
path = '/Users/saschastocker/Documents/data/images'
output = '/Users/saschastocker/Documents/data/imagesnormalised512'

#list of files in folder
files = os.listdir(path)
nii_files = [file for file in files if file.endswith('.nii.gz')]

print(f'lenght of files: {len(files)}')
print(f'files: {nii_files}')
for file in  nii_files:
    print(f'File coping: {file}')
    img_sitk = sitk.ReadImage(os.path.join(path, file))
    img_array = sitk.GetArrayFromImage(img_sitk)
    img_array = img_array.astype(np.float32)
    img_norm = normalize_3d_voxel(img_array)
    
    # Convert back to SimpleITK image and restore spatial information
    img_norm_sitk = sitk.GetImageFromArray(img_norm)
    img_norm_sitk.CopyInformation(img_sitk)  # Copy origin, spacing, direction metadata

    sitk.WriteImage(img_norm_sitk, os.path.join(output, file))

# %%

print('done')



