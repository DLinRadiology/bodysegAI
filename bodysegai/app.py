import os
import io
import glob
import time
import tempfile
import traceback
import threading
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
    load_nifti, load_single_dicom, save_nifti_mask, classify_uploads,
)
from .analysis import compute_areas, compute_mean_hu
from .visualization import hu_to_display, create_overlay_image, np_to_base64_png, create_single_tissue_overlay
from .pdf_export import generate_pdf

BASE_DIR = os.environ.get('BODYSEGAI_BASE',
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG_DIR = os.path.join(BASE_DIR, 'bodysegai') if os.environ.get('BODYSEGAI_BASE') else os.path.dirname(__file__)

app = Flask(__name__,
            template_folder=os.path.join(PKG_DIR, "templates"),
            static_folder=os.path.join(PKG_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Idle timeout (seconds) — server auto-shuts down after this much inactivity
IDLE_TIMEOUT = 1800  # 30 minutes

_last_activity = time.time()
_watchdog_started = False
_watchdog_lock = threading.Lock()


def _cleanup_temp_files():
    """Remove uploaded files and leftover temp NIfTI files."""
    for f in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, f))
        except OSError:
            pass
    for f in glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.nii.gz")):
        try:
            os.remove(f)
        except OSError:
            pass


def _idle_watchdog():
    """Background thread that exits the process after IDLE_TIMEOUT of inactivity."""
    while True:
        time.sleep(30)
        if time.time() - _last_activity > IDLE_TIMEOUT:
            print(f"BodySegAI: idle for {IDLE_TIMEOUT}s — shutting down.")
            _cleanup_temp_files()
            os._exit(0)


@app.before_request
def _track_activity():
    global _last_activity, _watchdog_started
    # Don't count idle-status polls as activity
    if request.path == "/api/idle-status":
        return
    _last_activity = time.time()
    with _watchdog_lock:
        if not _watchdog_started:
            _watchdog_started = True
            t = threading.Thread(target=_idle_watchdog, daemon=True)
            t.start()


# Session state (single-user desktop app)
state = {
    "loaded": False,
    "mode": None,
    "image_data": None,
    "slice_infos": [],       # per-slice: {filename, pixel_spacing, slice_thickness, source, affine}
    "hu_slices_oriented": [],
    "masks": [],
    "masks_original": None,
    "areas": [],
    "mean_hus": [],
    "patient_sex": "",
    "study_date": "",
    "segmented": False,
}


def _reset_state():
    state.update({
        "loaded": False, "mode": None, "image_data": None,
        "slice_infos": [],
        "hu_slices_oriented": [], "masks": [], "masks_original": None,
        "areas": [], "mean_hus": [],
        "patient_sex": "", "study_date": "",
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

        saved_paths = []
        for f in files:
            safe_name = f.filename.replace("/", "_").replace("\\", "_")
            path = os.path.join(UPLOAD_DIR, safe_name)
            f.save(path)
            saved_paths.append(path)

        file_type, paths = classify_uploads(saved_paths)

        # Load each file individually — never stack into a volume
        loaded = []  # list of img_info dicts (each single-slice)
        if file_type == "nifti":
            for p in paths:
                loaded.append(load_nifti(p))
        elif file_type in ("dicom_single", "dicom_series"):
            for p in paths:
                loaded.append(load_single_dicom(p))
        else:
            for p in paths:
                try:
                    loaded.append(load_single_dicom(p))
                except Exception:
                    return jsonify({"error": f"Unsupported file: {os.path.basename(p)}"}), 400

        # Build per-slice lists
        hu_slices = []
        slice_infos = []

        for img_info in loaded:
            data = img_info["data"]
            is_nifti = img_info["source"] == "nifti"
            info_base = {
                "filename": img_info["filename"],
                "pixel_spacing": img_info["pixel_spacing"],
                "slice_thickness": img_info["slice_thickness"],
                "source": img_info["source"],
                "affine": img_info.get("affine"),
            }

            if img_info["is_2d"] or len(data.shape) == 2:
                raw = data if len(data.shape) == 2 else data[:, :, 0]
                oriented = orient_for_display_nifti(raw) if is_nifti else orient_for_display_dicom(raw)
                hu_slices.append(oriented)
                slice_infos.append(info_base)
            else:
                # 3D NIfTI — expand into individual slices
                for i in range(data.shape[2]):
                    raw = data[:, :, i]
                    oriented = orient_for_display_nifti(raw) if is_nifti else orient_for_display_dicom(raw)
                    hu_slices.append(oriented)
                    slice_infos.append(info_base)

        mode = "single" if len(hu_slices) == 1 else "multi"

        # Use first file's patient info as default
        first = loaded[0]
        state.update({
            "loaded": True, "mode": mode, "image_data": first,
            "slice_infos": slice_infos,
            "hu_slices_oriented": hu_slices,
            "patient_sex": first["patient_sex"],
            "study_date": first["study_date"],
            "segmented": False, "masks": [], "masks_original": None,
            "areas": [], "mean_hus": [],
        })

        preview_idx = len(hu_slices) // 2 if mode == "multi" else 0
        display = hu_to_display(hu_slices[preview_idx])
        preview_b64 = np_to_base64_png(display)

        return jsonify({
            "success": True, "mode": mode, "n_slices": len(hu_slices),
            "preview": preview_b64,
            "patient_sex": state["patient_sex"],
            "study_date": state["study_date"],
            "slice_thickness": slice_infos[0]["slice_thickness"],
            "pixel_spacing": list(slice_infos[0]["pixel_spacing"]),
            "filename": slice_infos[0]["filename"],
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

        body = request.get_json() or {}
        state["patient_sex"] = body.get("patient_sex", state["patient_sex"])

        load_model()

        hu_slices = state["hu_slices_oriented"]
        slice_infos = state["slice_infos"]

        masks = []
        areas_list = []
        mean_hus_list = []

        for i in range(len(hu_slices)):
            hu_oriented = hu_slices[i]
            normalized = normalize_hu(hu_oriented)
            resized = resize_to_512(normalized)
            pred = predict_slice(resized)
            mask = process_prediction(pred, hu_oriented)
            masks.append(mask)
            areas_list.append(compute_areas(mask, slice_infos[i]["pixel_spacing"]))
            mean_hus_list.append(compute_mean_hu(mask, hu_oriented))

        state.update({
            "masks": masks,
            "areas": areas_list, "mean_hus": mean_hus_list,
            "segmented": True,
        })

        response = {
            "success": True, "mode": state["mode"],
            "n_slices": len(hu_slices), "slices": [],
        }

        for i in range(len(hu_slices)):
            display_hu = hu_to_display(hu_slices[i])
            overlay = create_overlay_image(display_hu, masks[i])

            slice_data = {
                "raw_preview": np_to_base64_png(display_hu),
                "overlay_preview": np_to_base64_png(overlay),
                "areas": {str(k): v for k, v in areas_list[i].items()},
                "mean_hu": {str(k): v for k, v in mean_hus_list[i].items()},
                "slice_index": i,
                "filename": slice_infos[i]["filename"],
                "slice_thickness": slice_infos[i]["slice_thickness"],
            }

            if state["mode"] == "single":
                tissue_overlays = {}
                for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_VAT, LABEL_SAT]:
                    timg = create_single_tissue_overlay(display_hu, masks[i], label)
                    tissue_overlays[str(label)] = np_to_base64_png(timg)
                slice_data["tissue_overlays"] = tissue_overlays

            response["slices"].append(slice_data)

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

    si = state["slice_infos"][0]
    pdf_bytes = generate_pdf(
        hu_image=state["hu_slices_oriented"][0],
        mask=state["masks"][0],
        areas=state["areas"][0],
        mean_hu=state["mean_hus"][0],
        patient_name=si["filename"],
        patient_sex=state["patient_sex"],
        study_date=state["study_date"],
        slice_thickness=si["slice_thickness"],
        pixel_spacing=si["pixel_spacing"],
        gender_code=gender_code,
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"bodysegai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    )


@app.route("/api/download/excel")
def download_excel():
    """Download Excel (.xlsx) with areas and HU values."""
    if not state["segmented"]:
        return jsonify({"error": "No segmentation available"}), 400

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Body Composition"

    headers = [
        "Slice", "Image Name",
        "SM Area (cm2)", "IMAT Area (cm2)", "VAT Area (cm2)", "SAT Area (cm2)",
        "SM Mean HU", "IMAT Mean HU", "VAT Mean HU", "SAT Mean HU",
        "Slice Thickness (mm)",
    ]
    ws.append(headers)

    # Bold headers
    from openpyxl.styles import Font
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for i in range(len(state["areas"])):
        si = state["slice_infos"][i] if i < len(state["slice_infos"]) else {}
        a = state["areas"][i]
        h = state["mean_hus"][i]
        ws.append([
            i + 1, si.get("filename", "unknown"),
            a.get(LABEL_MUSCLE, 0), a.get(LABEL_IMAT, 0),
            a.get(LABEL_VAT, 0), a.get(LABEL_SAT, 0),
            h.get(LABEL_MUSCLE, ""), h.get(LABEL_IMAT, ""),
            h.get(LABEL_VAT, ""), h.get(LABEL_SAT, ""),
            si.get("slice_thickness"),
        ])

    # Auto-width columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 2

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"bodysegai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    )


@app.route("/api/download/mask")
def download_mask():
    """Download segmentation mask(s) as .nii.gz (single) or .zip (multi)."""
    if not state["segmented"]:
        return jsonify({"error": "No segmentation available"}), 400

    import zipfile

    masks = state["masks"]
    slice_infos = state["slice_infos"]

    def _mask_filename(si):
        base = si.get("filename", "output")
        for ext in [".nii.gz", ".nii", ".dcm", ".gz"]:
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
                break
        return base

    if len(masks) == 1:
        # Single mask — save as .nii.gz
        si = slice_infos[0]
        is_nifti = si["source"] == "nifti"
        mask_data = undo_orientation_nifti(masks[0]) if is_nifti else masks[0]

        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp.close()
        save_nifti_mask(mask_data, si.get("affine"), tmp.name)

        return send_file(
            tmp.name,
            mimetype="application/gzip",
            as_attachment=True,
            download_name=f"mask_{_mask_filename(si)}.nii.gz",
        )
    else:
        # Multiple masks — zip individual .nii.gz files
        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp_zip.close()

        with zipfile.ZipFile(tmp_zip.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (mask, si) in enumerate(zip(masks, slice_infos)):
                is_nifti = si["source"] == "nifti"
                mask_data = undo_orientation_nifti(mask) if is_nifti else mask

                tmp_nii = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                tmp_nii.close()
                save_nifti_mask(mask_data, si.get("affine"), tmp_nii.name)

                arcname = f"mask_{_mask_filename(si)}.nii.gz"
                zf.write(tmp_nii.name, arcname)
                os.remove(tmp_nii.name)

        return send_file(
            tmp_zip.name,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"bodysegai_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
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

        # Determine extension for saving
        fname = f.filename or "mask.nii.gz"
        if fname.endswith(".nii.gz"):
            save_name = "corrected_mask.nii.gz"
        else:
            save_name = "corrected_mask.nii"

        path = os.path.join(UPLOAD_DIR, save_name)
        f.save(path)

        import nibabel as nib
        mask_img = nib.load(path)
        mask_data = mask_img.get_fdata()

        img_info = state["image_data"]
        is_nifti = img_info["source"] == "nifti"
        hu_slices = state["hu_slices_oriented"]

        masks = []
        if len(mask_data.shape) == 2 or (len(mask_data.shape) == 3 and mask_data.shape[2] == 1):
            raw_mask = mask_data if len(mask_data.shape) == 2 else mask_data[:, :, 0]
            oriented = orient_for_display_nifti(raw_mask) if is_nifti else raw_mask
            masks.append(oriented.astype(np.uint8))
        else:
            for i in range(mask_data.shape[2]):
                oriented = orient_for_display_nifti(mask_data[:, :, i]) if is_nifti else mask_data[:, :, i]
                masks.append(oriented.astype(np.uint8))

        areas_list = []
        mean_hus_list = []
        slice_infos = state["slice_infos"]
        for i in range(len(masks)):
            ps = slice_infos[i]["pixel_spacing"] if i < len(slice_infos) else (1.0, 1.0)
            areas_list.append(compute_areas(masks[i], ps))
            mean_hus_list.append(compute_mean_hu(masks[i], hu_slices[i]))

        state.update({
            "masks": masks,
            "areas": areas_list, "mean_hus": mean_hus_list,
            "segmented": True,
        })

        response = {
            "success": True, "mode": state["mode"],
            "n_slices": len(masks), "slices": [],
        }

        for i in range(len(masks)):
            display_hu = hu_to_display(hu_slices[i])
            overlay = create_overlay_image(display_hu, masks[i])

            si = slice_infos[i] if i < len(slice_infos) else {}
            slice_data = {
                "raw_preview": np_to_base64_png(display_hu),
                "overlay_preview": np_to_base64_png(overlay),
                "areas": {str(k): v for k, v in areas_list[i].items()},
                "mean_hu": {str(k): v for k, v in mean_hus_list[i].items()},
                "slice_index": i,
                "filename": si.get("filename", ""),
                "slice_thickness": si.get("slice_thickness"),
            }

            if state["mode"] == "single":
                tissue_overlays = {}
                for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_VAT, LABEL_SAT]:
                    timg = create_single_tissue_overlay(display_hu, masks[i], label)
                    tissue_overlays[str(label)] = np_to_base64_png(timg)
                slice_data["tissue_overlays"] = tissue_overlays

            response["slices"].append(slice_data)

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/licence")
def serve_licence():
    """Serve the licence PDF."""
    licence_path = os.path.join(BASE_DIR, "licence.pdf")
    if os.path.exists(licence_path):
        return send_file(licence_path, mimetype="application/pdf")
    return jsonify({"error": "Licence file not found"}), 404


@app.route("/api/keepalive")
def keepalive():
    """Reset the idle timer (called by frontend 'Stay alive' button)."""
    global _last_activity
    _last_activity = time.time()
    return jsonify({"ok": True})


@app.route("/api/idle-status")
def idle_status():
    """Return how long the server has been idle and when it will shut down."""
    idle = time.time() - _last_activity
    return jsonify({
        "idle_seconds": round(idle),
        "timeout_seconds": IDLE_TIMEOUT,
        "remaining_seconds": max(0, round(IDLE_TIMEOUT - idle)),
    })


@app.route("/api/shutdown")
def shutdown():
    """Shut down the application."""
    def _shutdown():
        time.sleep(0.5)
        _cleanup_temp_files()
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"message": "Shutting down..."})
