import numpy as np
from .postprocessing import LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT, TISSUE_NAMES


def compute_areas(mask: np.ndarray, pixel_spacing: tuple) -> dict:
    """
    Compute tissue areas in cm² from a labeled mask.
    pixel_spacing = (row_spacing_mm, col_spacing_mm)
    """
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    pixel_area_cm2 = pixel_area_mm2 / 100.0

    areas = {}
    for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
        count = int(np.sum(mask == label))
        areas[label] = round(count * pixel_area_cm2, 2)
    return areas


def compute_mean_hu(mask: np.ndarray, hu_image: np.ndarray) -> dict:
    """Compute mean HU for each tissue class."""
    mean_hu = {}
    for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
        pixels = hu_image[mask == label]
        if len(pixels) > 0:
            mean_hu[label] = round(float(np.mean(pixels)), 1)
        else:
            mean_hu[label] = None
    return mean_hu


# Reference values for L3 single-slice body composition
# These are among the most cited in the literature
REFERENCES = {
    "M": {
        LABEL_MUSCLE: {
            "name": "Skeletal Muscle Index (SMI)",
            "note": "Area shown. SMI cutoff requires height².",
            "cutoff_smi": 52.4,  # cm²/m², Prado et al. 2008
            "source": "Prado et al. Lancet Oncol 2008",
        },
        LABEL_VAT: {
            "name": "Visceral Adipose Tissue",
            "cutoff_area": 163.8,  # cm²
            "source": "Doyle et al. Obesity 2013",
            "note": "Area >163.8 cm² associated with metabolic syndrome.",
        },
    },
    "F": {
        LABEL_MUSCLE: {
            "name": "Skeletal Muscle Index (SMI)",
            "note": "Area shown. SMI cutoff requires height².",
            "cutoff_smi": 38.5,  # cm²/m², Prado et al. 2008
            "source": "Prado et al. Lancet Oncol 2008",
        },
        LABEL_VAT: {
            "name": "Visceral Adipose Tissue",
            "cutoff_area": 80.1,  # cm²
            "source": "Doyle et al. Obesity 2013",
            "note": "Area >80.1 cm² associated with metabolic syndrome.",
        },
    },
}


def get_reference_check(areas: dict, gender: str) -> dict:
    """
    Check areas against reference values.
    Returns dict with label -> {status, message, source}
    gender should be 'M' or 'F'
    """
    if gender not in REFERENCES:
        return {}

    refs = REFERENCES[gender]
    results = {}

    for label, ref in refs.items():
        area = areas.get(label, 0)
        result = {"source": ref["source"], "note": ref.get("note", "")}

        if "cutoff_area" in ref:
            cutoff = ref["cutoff_area"]
            if area > cutoff:
                result["status"] = "high"
                result["message"] = f"{area:.1f} cm² (>{cutoff} cm²)"
            else:
                result["status"] = "normal"
                result["message"] = f"{area:.1f} cm² (≤{cutoff} cm²)"
        else:
            result["status"] = "info"
            result["message"] = ref.get("note", "")

        results[label] = result

    return results
