import os
import io
import csv
import tempfile
import traceback
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

from .model import load_model, predict_slice
from .preprocessing import (
    normalize_hu, prepare_for_model_nifti, prepare_for_model_dicom,
    resize_to_512, orient_for_display_nifti, orient_for_display_dicom,
    undo_orientation_nifti,
)
from .postprocessing import (
    process_prediction, LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT, TISSUE_NAMES,
)
from .io_handlers import (
    load_nifti, load_single_dicom, load_dicom_series, save_nifti_mask, classify_uploads,
)
from .analysis import compute_areas, compute_mean_hu, get_reference_check
from .visualization import hu_to_display, create_overlay_image, np_to_base64_png, create_single_tissue_overlay
from .pdf_export import generate_pdf

BASE_DIR = os.environ.get('BODYSEGAI_BASE',
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# For templates/static, use the package dir (PyInstaller copies these into _MEIPASS)
PKG_DIR = os.path.join(BASE_DIR, 'bodysegai') if os.environ.get('BODYSEGAI_BASE') else os.path.dirname(__file__)

app = Flask(__name__,
            template_folder=os.path.join(PKG_DIR, "templates"),
            static_folder=os.path.join(PKG_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session state (single-user desktop app)
state = {
    "loaded": False,
    "mode": None,  # "single" or "multi"
    "image_data": None,  # dict from io_handlers
    "hu_slices_oriented": [],  # list of oriented HU slices
    "masks": [],  # list of mask arrays (oriented for display)
    "masks_original": None,  # mask in original space (for saving)
    "areas": [],  # list of area dicts per slice
    "mean_hus": [],  # list of mean_hu dicts per slice
    "patient_name": "",
    "patient_sex": "",
    "study_date": "",
    "pixel_spacing": (1.0, 1.0),
    "slice_thickness": None,
    "segmented": False,
}


def _reset_state():
    state.update({
        "loaded": False, "mode": None, "image_data": None,
        "hu_slices_oriented": [], "masks": [], "masks_original": None,
        "areas": [], "mean_hus": [],
        "patient_name": "", "patient_sex": "", "study_date": "",
        "pixel_spacing": (1.0, 1.0), "slice_thickness": None,
        "segmented": False,
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Handle file upload(s). Returns preview info."""
    try:
        _reset_state()

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        # Save uploaded files to temp dir
        saved_paths = []
        for f in files:
            safe_name = f.filename.replace("/", "_").replace("\\", "_")
            path = os.path.join(UPLOAD_DIR, safe_name)
            f.save(path)
            saved_paths.append(path)

        # Classify and load
        file_type, paths = classify_uploads(saved_paths)

        if file_type == "nifti":
            img_info = load_nifti(paths[0])
        elif file_type == "dicom_single":
            img_info = load_single_dicom(paths[0])
        elif file_type == "dicom_series":
            img_info = load_dicom_series(paths)
        else:
            # Try as DICOM
            try:
                if len(paths) == 1:
                    img_info = load_single_dicom(paths[0])
                else:
                    img_info = load_dicom_series(paths)
            except Exception:
                return jsonify({"error": "Unsupported file format"}), 400

        data = img_info["data"]
        is_nifti = img_info["source"] == "nifti"

        # Determine mode
        if img_info["is_2d"] or img_info["n_slices"] == 1:
            mode = "single"
        else:
            mode = "multi"

        # Orient slices for display
        hu_slices = []
        if img_info["is_2d"] or len(data.shape) == 2:
            raw = data if len(data.shape) == 2 else data[:, :, 0]
            oriented = orient_for_display_nifti(raw) if is_nifti else orient_for_display_dicom(raw)
            hu_slices.append(oriented)
        else:
            for i in range(data.shape[2]):
                raw = data[:, :, i]
                oriented = orient_for_display_nifti(raw) if is_nifti else orient_for_display_dicom(raw)
                hu_slices.append(oriented)

        state.update({
            "loaded": True,
            "mode": mode,
            "image_data": img_info,
            "hu_slices_oriented": hu_slices,
            "patient_name": img_info["patient_name"],
            "patient_sex": img_info["patient_sex"],
            "study_date": img_info["study_date"],
            "pixel_spacing": img_info["pixel_spacing"],
            "slice_thickness": img_info["slice_thickness"],
            "segmented": False,
            "masks": [],
            "masks_original": None,
            "areas": [],
            "mean_hus": [],
        })

        # Generate preview (middle slice for multi, the slice for single)
        preview_idx = len(hu_slices) // 2 if mode == "multi" else 0
        display = hu_to_display(hu_slices[preview_idx])
        preview_b64 = np_to_base64_png(display)

        return jsonify({
            "success": True,
            "mode": mode,
            "n_slices": len(hu_slices),
            "preview": preview_b64,
            "patient_name": state["patient_name"],
            "patient_sex": state["patient_sex"],
            "study_date": state["study_date"],
            "slice_thickness": state["slice_thickness"],
            "pixel_spacing": list(state["pixel_spacing"]),
            "filename": img_info["filename"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/segment", methods=["POST"])
def segment():
    """Run segmentation on loaded data."""
    try:
        if not state["loaded"]:
            return jsonify({"error": "No image loaded"}), 400

        # Update patient info from request
        body = request.get_json() or {}
        state["patient_name"] = body.get("patient_name", state["patient_name"])
        state["patient_sex"] = body.get("patient_sex", state["patient_sex"])

        load_model()

        img_info = state["image_data"]
        data = img_info["data"]
        is_nifti = img_info["source"] == "nifti"
        hu_slices = state["hu_slices_oriented"]

        masks = []
        areas_list = []
        mean_hus_list = []

        n_slices = len(hu_slices)

        for i in range(n_slices):
            hu_oriented = hu_slices[i]

            # Preprocess for model
            normalized = normalize_hu(hu_oriented)
            resized = resize_to_512(normalized)

            # Predict
            pred = predict_slice(resized)

            # Postprocess
            mask = process_prediction(pred, hu_oriented)

            masks.append(mask)

            # Compute areas and HU
            areas = compute_areas(mask, state["pixel_spacing"])
            mean_hu = compute_mean_hu(mask, hu_oriented)
            areas_list.append(areas)
            mean_hus_list.append(mean_hu)

        # Build original-space mask for NIfTI saving
        if is_nifti:
            orig_data = img_info["data"]
            if img_info["is_2d"] or len(orig_data.shape) == 2:
                masks_orig = undo_orientation_nifti(masks[0])
            else:
                masks_orig = np.zeros(orig_data.shape, dtype=np.uint8)
                for i, m in enumerate(masks):
                    masks_orig[:, :, i] = undo_orientation_nifti(m)
        else:
            # DICOM: no rotation needed, just stack
            if len(masks) == 1:
                masks_orig = masks[0]
            else:
                masks_orig = np.stack(masks, axis=-1)

        state.update({
            "masks": masks,
            "masks_original": masks_orig,
            "areas": areas_list,
            "mean_hus": mean_hus_list,
            "segmented": True,
        })

        # Build response
        response = {
            "success": True,
            "mode": state["mode"],
            "n_slices": n_slices,
            "slices": [],
        }

        for i in range(n_slices):
            display_hu = hu_to_display(hu_slices[i])
            overlay = create_overlay_image(display_hu, masks[i])

            slice_data = {
                "raw_preview": np_to_base64_png(display_hu),
                "overlay_preview": np_to_base64_png(overlay),
                "areas": {str(k): v for k, v in areas_list[i].items()},
                "mean_hu": {str(k): v for k, v in mean_hus_list[i].items()},
                "slice_index": i,
            }

            # For single mode, add per-tissue overlays
            if state["mode"] == "single":
                tissue_overlays = {}
                for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
                    timg = create_single_tissue_overlay(display_hu, masks[i], label)
                    tissue_overlays[str(label)] = np_to_base64_png(timg)
                slice_data["tissue_overlays"] = tissue_overlays

            response["slices"].append(slice_data)

        # Reference check (single mode only)
        if state["mode"] == "single" and state["patient_sex"]:
            gender = state["patient_sex"][0].upper() if state["patient_sex"] else ""
            ref_check = get_reference_check(areas_list[0], gender)
            response["references"] = {str(k): v for k, v in ref_check.items()}

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/pdf")
def download_pdf():
    """Download PDF report (single slice mode)."""
    if not state["segmented"] or state["mode"] != "single":
        return jsonify({"error": "No single-slice segmentation available"}), 400

    gender_code = ""
    if state["patient_sex"]:
        gender_code = state["patient_sex"][0].upper()

    pdf_bytes = generate_pdf(
        hu_image=state["hu_slices_oriented"][0],
        mask=state["masks"][0],
        areas=state["areas"][0],
        mean_hu=state["mean_hus"][0],
        patient_name=state["patient_name"],
        patient_sex=state["patient_sex"],
        study_date=state["study_date"],
        slice_thickness=state["slice_thickness"],
        pixel_spacing=state["pixel_spacing"],
        gender_code=gender_code,
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"bodysegai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    )


@app.route("/api/download/csv")
def download_csv():
    """Download CSV with areas and HU values."""
    if not state["segmented"]:
        return jsonify({"error": "No segmentation available"}), 400

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Slice", "Image Name",
        "SM Area (cm2)", "IMAT Area (cm2)", "SAT Area (cm2)", "VAT Area (cm2)",
        "SM Mean HU", "IMAT Mean HU", "SAT Mean HU", "VAT Mean HU",
    ])

    filename = state["image_data"]["filename"] if state["image_data"] else "unknown"
    for i in range(len(state["areas"])):
        a = state["areas"][i]
        h = state["mean_hus"][i]
        writer.writerow([
            i + 1, filename,
            a.get(LABEL_MUSCLE, 0), a.get(LABEL_IMAT, 0),
            a.get(LABEL_SAT, 0), a.get(LABEL_VAT, 0),
            h.get(LABEL_MUSCLE, ""), h.get(LABEL_IMAT, ""),
            h.get(LABEL_SAT, ""), h.get(LABEL_VAT, ""),
        ])

    output = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"bodysegai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


@app.route("/api/download/mask")
def download_mask():
    """Download segmentation mask as .nii.gz."""
    if not state["segmented"]:
        return jsonify({"error": "No segmentation available"}), 400

    affine = state["image_data"].get("affine") if state["image_data"] else None
    mask_data = state["masks_original"]

    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    save_nifti_mask(mask_data, affine, tmp.name)

    return send_file(
        tmp.name,
        mimetype="application/gzip",
        as_attachment=True,
        download_name=f"bodysegai_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nii.gz",
    )


@app.route("/api/upload_mask", methods=["POST"])
def upload_corrected_mask():
    """Upload a corrected mask, recompute areas and HU."""
    try:
        if not state["loaded"]:
            return jsonify({"error": "No image loaded"}), 400

        f = request.files.get("mask")
        if not f:
            return jsonify({"error": "No mask file"}), 400

        path = os.path.join(UPLOAD_DIR, "corrected_mask.nii.gz")
        f.save(path)

        import nibabel as nib
        mask_img = nib.load(path)
        mask_data = mask_img.get_fdata()

        img_info = state["image_data"]
        is_nifti = img_info["source"] == "nifti"
        hu_slices = state["hu_slices_oriented"]

        # Orient the corrected mask for display
        masks = []
        if len(mask_data.shape) == 2 or (len(mask_data.shape) == 3 and mask_data.shape[2] == 1):
            raw_mask = mask_data if len(mask_data.shape) == 2 else mask_data[:, :, 0]
            oriented = orient_for_display_nifti(raw_mask) if is_nifti else raw_mask
            masks.append(oriented.astype(np.uint8))
        else:
            for i in range(mask_data.shape[2]):
                oriented = orient_for_display_nifti(mask_data[:, :, i]) if is_nifti else mask_data[:, :, i]
                masks.append(oriented.astype(np.uint8))

        # Recompute areas and HU
        areas_list = []
        mean_hus_list = []
        for i in range(len(masks)):
            areas = compute_areas(masks[i], state["pixel_spacing"])
            mean_hu = compute_mean_hu(masks[i], hu_slices[i])
            areas_list.append(areas)
            mean_hus_list.append(mean_hu)

        state.update({
            "masks": masks,
            "masks_original": mask_data,
            "areas": areas_list,
            "mean_hus": mean_hus_list,
            "segmented": True,
        })

        # Build response (same format as segment)
        response = {
            "success": True,
            "mode": state["mode"],
            "n_slices": len(masks),
            "slices": [],
        }

        for i in range(len(masks)):
            display_hu = hu_to_display(hu_slices[i])
            overlay = create_overlay_image(display_hu, masks[i])

            slice_data = {
                "raw_preview": np_to_base64_png(display_hu),
                "overlay_preview": np_to_base64_png(overlay),
                "areas": {str(k): v for k, v in areas_list[i].items()},
                "mean_hu": {str(k): v for k, v in mean_hus_list[i].items()},
                "slice_index": i,
            }

            if state["mode"] == "single":
                tissue_overlays = {}
                for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
                    timg = create_single_tissue_overlay(display_hu, masks[i], label)
                    tissue_overlays[str(label)] = np_to_base64_png(timg)
                slice_data["tissue_overlays"] = tissue_overlays

            response["slices"].append(slice_data)

        if state["mode"] == "single" and state["patient_sex"]:
            gender = state["patient_sex"][0].upper()
            ref_check = get_reference_check(areas_list[0], gender)
            response["references"] = {str(k): v for k, v in ref_check.items()}

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Need this import for upload_corrected_mask
from .preprocessing import orient_for_display_nifti
