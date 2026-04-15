import numpy as np
from skimage.transform import resize

# Normalization constants from training
HU_MEAN = 535.7372827952495
HU_STD = 492.83128067388367
HU_MIN = -400
HU_MAX = 600


def normalize_hu(image: np.ndarray) -> np.ndarray:
    """Clip HU range and z-score normalize."""
    image = image.copy().astype(np.float64)
    image[image < HU_MIN] = HU_MIN
    image[image > HU_MAX] = HU_MAX
    image = (image + HU_MEAN) / HU_STD
    return image


def prepare_for_model_nifti(slice_2d: np.ndarray) -> np.ndarray:
    """Apply rotation/flip needed for NIfTI data (dcm2nii convention)."""
    out = np.rot90(slice_2d, k=3)
    out = np.fliplr(out)
    return out


def prepare_for_model_dicom(slice_2d: np.ndarray) -> np.ndarray:
    """DICOM data: no rotation needed (already in radiological orientation)."""
    return slice_2d


def resize_to_512(image: np.ndarray) -> np.ndarray:
    if image.shape != (512, 512):
        return resize(image, (512, 512), anti_aliasing=False,
                       preserve_range=True, mode="constant")
    return image


def orient_for_display_nifti(slice_2d: np.ndarray) -> np.ndarray:
    """Orient a raw NIfTI slice for radiological display (same as model prep)."""
    out = np.rot90(slice_2d, k=3)
    out = np.fliplr(out)
    return out


def orient_for_display_dicom(slice_2d: np.ndarray) -> np.ndarray:
    """DICOM is already in radiological orientation."""
    return slice_2d


def undo_orientation_nifti(slice_2d: np.ndarray) -> np.ndarray:
    """Reverse the NIfTI orientation transform for saving back."""
    out = np.fliplr(slice_2d)
    out = np.rot90(out, k=1)
    return out
