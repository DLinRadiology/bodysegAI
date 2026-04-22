import numpy as np
from PIL import Image
import io
import base64

from .postprocessing import LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT

# Tissue colors (RGB)
TISSUE_COLORS = {
    LABEL_MUSCLE: (239, 68, 68),     # Red
    LABEL_IMAT: (57, 255, 20),       # Neon green
    LABEL_VAT: (250, 204, 21),       # Yellow
    LABEL_SAT: (37, 245, 252),       # Light blue
}

# CT window for body composition display
WINDOW_CENTER = 40
WINDOW_WIDTH = 400


def hu_to_display(hu_image, wc=WINDOW_CENTER, ww=WINDOW_WIDTH):
    """Convert HU image to 0-255 display range using windowing."""
    lo = wc - ww / 2
    hi = wc + ww / 2
    img = np.clip(hu_image, lo, hi)
    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    return img


def grayscale_to_rgb(gray):
    """Convert single-channel uint8 to 3-channel RGB."""
    return np.stack([gray, gray, gray], axis=-1)


def create_overlay_image(display_gray, mask, alpha=1.0):
    """Create RGB image with all tissue classes overlaid on grayscale CT."""
    rgb = grayscale_to_rgb(display_gray).astype(np.float64)

    for label, color in TISSUE_COLORS.items():
        region = mask == label
        if np.any(region):
            for c in range(3):
                rgb[:, :, c][region] = (
                    rgb[:, :, c][region] * (1 - alpha) + color[c] * alpha
                )

    return np.clip(rgb, 0, 255).astype(np.uint8)


def create_single_tissue_overlay(display_gray, mask, label, alpha=1.0):
    """Create RGB image with just one tissue class highlighted."""
    rgb = grayscale_to_rgb(display_gray).astype(np.float64)
    color = TISSUE_COLORS.get(label, (255, 255, 255))
    region = mask == label

    if np.any(region):
        for c in range(3):
            rgb[:, :, c][region] = (
                rgb[:, :, c][region] * (1 - alpha) + color[c] * alpha
            )

    return np.clip(rgb, 0, 255).astype(np.uint8)


def np_to_base64_png(np_img):
    """Convert numpy RGB/grayscale image to base64 PNG string."""
    if np_img.ndim == 2:
        pil = Image.fromarray(np_img, mode="L")
    else:
        pil = Image.fromarray(np_img, mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
