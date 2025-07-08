import nibabel as nib
import numpy as np
import os

def load_fmri_embeddings(subject_dir, words):
    """
    Extract averaged fMRI activations corresponding to specific words.
    Assumes data organized in BIDS format with preprocessed betas or task activations.
    """
    # For simplicity, we fake word->brain pattern mapping:
    # Use the first N volumes of a single subject (1 per word)

    bold_path = os.path.join(subject_dir, "func", "sub-01_task-listening_run-1_bold.nii.gz")
    if not os.path.exists(bold_path):
        raise FileNotFoundError(f"Missing file: {bold_path}")
    
    img = nib.load(bold_path)
    data = img.get_fdata()  # shape: (x, y, z, time)

    # Flatten spatial dims and take one timepoint per word
    flattened = data.reshape(-1, data.shape[-1])  # (voxels, time)
    selected = flattened[:, :len(words)].T        # (words, voxels)

    return selected
