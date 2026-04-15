import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom


def load_nifti(filepath):
    """Load a NIfTI file. Returns dict with image data and metadata."""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Pixel spacing from header
    pixdim = header.get_zooms()
    pixel_spacing = (float(pixdim[0]), float(pixdim[1])) if len(pixdim) >= 2 else (1.0, 1.0)
    slice_thickness = float(pixdim[2]) if len(pixdim) >= 3 else None

    is_2d = len(data.shape) == 2
    if is_2d:
        n_slices = 1
    else:
        n_slices = data.shape[2]

    return {
        "data": data,
        "affine": affine,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "n_slices": n_slices,
        "is_2d": is_2d,
        "source": "nifti",
        "filename": os.path.basename(filepath),
        "patient_name": "",
        "patient_sex": "",
        "study_date": "",
    }


def load_single_dicom(filepath):
    """Load a single DICOM file. Returns dict with image data and metadata."""
    # Use SimpleITK for pixel data (handles all transfer syntaxes)
    sitk_img = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(sitk_img)

    # Squeeze out extra dimensions: SimpleITK may return (1, H, W)
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]

    # Use pydicom for metadata
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)

    pixel_spacing = _get_pixel_spacing(ds)
    slice_thickness = _get_slice_thickness(ds)
    patient_name = str(getattr(ds, "PatientName", "")) or ""
    patient_sex = str(getattr(ds, "PatientSex", "")) or ""
    study_date = str(getattr(ds, "StudyDate", "")) or ""

    return {
        "data": data.astype(np.float64),
        "affine": None,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "n_slices": 1,
        "is_2d": True,
        "source": "dicom",
        "filename": os.path.basename(filepath),
        "patient_name": patient_name,
        "patient_sex": patient_sex,
        "study_date": study_date,
    }


def load_dicom_series(file_list):
    """Load multiple DICOM files as a sorted volume. Returns dict."""
    # Read all files, sort by InstanceNumber or SliceLocation
    slices_info = []
    for fp in file_list:
        ds = pydicom.dcmread(fp, stop_before_pixels=True)
        instance_num = getattr(ds, "InstanceNumber", 0) or 0
        slice_loc = getattr(ds, "SliceLocation", 0) or 0
        slices_info.append((fp, int(instance_num), float(slice_loc)))

    # Sort by instance number, then slice location
    slices_info.sort(key=lambda x: (x[1], x[2]))

    # Load pixel data
    arrays = []
    for fp, _, _ in slices_info:
        sitk_img = sitk.ReadImage(fp)
        arr = sitk.GetArrayFromImage(sitk_img)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        arrays.append(arr)

    # Stack into volume (H, W, N_slices)
    volume = np.stack(arrays, axis=-1).astype(np.float64)

    # Metadata from first file
    ds = pydicom.dcmread(slices_info[0][0], stop_before_pixels=True)
    pixel_spacing = _get_pixel_spacing(ds)
    slice_thickness = _get_slice_thickness(ds)

    return {
        "data": volume,
        "affine": None,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "n_slices": len(arrays),
        "is_2d": False,
        "source": "dicom",
        "filename": os.path.basename(slices_info[0][0]),
        "patient_name": str(getattr(ds, "PatientName", "")) or "",
        "patient_sex": str(getattr(ds, "PatientSex", "")) or "",
        "study_date": str(getattr(ds, "StudyDate", "")) or "",
    }


def save_nifti_mask(mask_data, affine, filepath):
    """Save a mask array as NIfTI (.nii.gz)."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(mask_data.astype(np.float32), affine)
    nib.save(img, filepath)


def _get_pixel_spacing(ds):
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None:
        return (float(ps[0]), float(ps[1]))
    ips = getattr(ds, "ImagerPixelSpacing", None)
    if ips is not None:
        return (float(ips[0]), float(ips[1]))
    return (1.0, 1.0)


def _get_slice_thickness(ds):
    st = getattr(ds, "SliceThickness", None)
    if st is not None:
        return float(st)
    return None


def classify_uploads(file_paths):
    """Classify a list of uploaded file paths into NIfTI or DICOM groups.
    Returns (file_type, paths) where file_type is 'nifti', 'dicom_single', or 'dicom_series'."""
    nifti_files = [f for f in file_paths if f.endswith(('.nii', '.nii.gz'))]
    dicom_files = [f for f in file_paths if f.endswith('.dcm') or _is_dicom(f)]

    if nifti_files:
        return ("nifti", nifti_files)
    elif len(dicom_files) == 1:
        return ("dicom_single", dicom_files)
    elif len(dicom_files) > 1:
        return ("dicom_series", dicom_files)
    else:
        # Try to detect file type
        return ("unknown", file_paths)


def _is_dicom(filepath):
    """Check if a file is DICOM by reading its magic bytes."""
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False
