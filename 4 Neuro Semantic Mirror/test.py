import nibabel as nib
import numpy as np

# Load the NIfTI image
nifti_file_path = r'ds004301-1.0.2\derivatives\preprocessed_data\sub-01\func\sub-01_task-listening_run-1_bold.nii.gz'
img = nib.load(nifti_file_path)

# Get the image data as a NumPy array
data = img.get_fdata()

# Print the shape of the data
print(data.shape)

# Example: Access a specific voxel
voxel_value = data[0, 0, 0]  # Access the first voxel
print(voxel_value)
