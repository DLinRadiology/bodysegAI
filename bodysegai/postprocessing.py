import numpy as np
from skimage.transform import resize

# Label map
LABEL_BACKGROUND = 0
LABEL_MUSCLE = 1
LABEL_IMAT = 2
LABEL_SAT = 5
LABEL_VAT = 7

TISSUE_NAMES = {
    LABEL_MUSCLE: "Skeletal Muscle",
    LABEL_IMAT: "Intramuscular Adipose Tissue (IMAT)",
    LABEL_SAT: "Subcutaneous Adipose Tissue (SAT)",
    LABEL_VAT: "Visceral Adipose Tissue (VAT)",
}

# HU thresholds per channel
# Channel 0 = muscle, Channel 1 = SAT, Channel 2 = VAT
CHANNEL_HU_RANGES = {
    0: (-190, 150),   # muscle
    1: (-150, -50),   # SAT
    2: (-190, -30),   # VAT
}

IMAT_HU_THRESHOLD = -30


def process_prediction(pred: np.ndarray, orig_hu: np.ndarray) -> np.ndarray:
    """
    Process raw model output (H,W,3) + original HU values (H,W) into
    a labeled mask (H,W) with values 0,1,2,5,7.

    pred and orig_hu must be in the same orientation and spatial dimensions.
    """
    orig_shape = orig_hu.shape

    # Resize prediction to match original if needed
    if pred.shape[:2] != orig_shape:
        resized = np.zeros((orig_shape[0], orig_shape[1], 3))
        for i in range(3):
            resized[:, :, i] = resize(pred[:, :, i], orig_shape,
                                       anti_aliasing=False,
                                       preserve_range=True, mode="constant")
        pred = resized

    # Apply HU-based masking per channel
    for ch, (hu_min, hu_max) in CHANNEL_HU_RANGES.items():
        pred[:, :, ch][orig_hu < hu_min] = 0
        pred[:, :, ch][orig_hu > hu_max] = 0

    # Winner-takes-all across channels
    maximum = np.max(pred, axis=2)
    maximum[maximum < 0.5] = -1

    mask = np.zeros(orig_shape, dtype=np.uint8)
    mask[pred[:, :, 0] == maximum] = LABEL_MUSCLE
    # IMAT: muscle pixels with low HU
    mask[(mask == LABEL_MUSCLE) & (orig_hu <= IMAT_HU_THRESHOLD)] = LABEL_IMAT
    mask[pred[:, :, 1] == maximum] = LABEL_SAT
    mask[pred[:, :, 2] == maximum] = LABEL_VAT

    return mask
