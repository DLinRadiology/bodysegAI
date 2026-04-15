import io
import os
import numpy as np
from PIL import Image
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from .postprocessing import LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT, TISSUE_NAMES
from .analysis import REFERENCES
from .visualization import create_overlay_image, create_single_tissue_overlay, hu_to_display

# Colors matching the web UI
TISSUE_COLORS_RGB = {
    LABEL_MUSCLE: (239, 68, 68),
    LABEL_IMAT: (245, 158, 11),
    LABEL_SAT: (59, 130, 246),
    LABEL_VAT: (16, 185, 129),
}

TISSUE_COLORS_HEX = {
    LABEL_MUSCLE: "#EF4444",
    LABEL_IMAT: "#F59E0B",
    LABEL_SAT: "#3B82F6",
    LABEL_VAT: "#10B981",
}


def _np_to_rl_image(np_img, width_cm=7):
    """Convert numpy RGB image to a ReportLab Image object."""
    pil = Image.fromarray(np_img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    aspect = pil.height / pil.width
    w = width_cm * cm
    h = w * aspect
    return RLImage(buf, width=w, height=h)


def generate_pdf(
    hu_image,
    mask,
    areas,
    mean_hu,
    patient_name="",
    patient_sex="",
    study_date="",
    slice_thickness=None,
    pixel_spacing=(1.0, 1.0),
    gender_code="",
):
    """Generate a PDF report. Returns bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2",
                              fontName="Helvetica-Bold", fontSize=18,
                              textColor=HexColor("#1E293B"),
                              spaceAfter=6*mm, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Subtitle",
                              fontName="Helvetica", fontSize=10,
                              textColor=HexColor("#64748B"),
                              spaceAfter=4*mm, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="SectionHead",
                              fontName="Helvetica-Bold", fontSize=12,
                              textColor=HexColor("#1E293B"),
                              spaceBefore=6*mm, spaceAfter=3*mm))
    styles.add(ParagraphStyle(name="BodySmall",
                              fontName="Helvetica", fontSize=9,
                              textColor=HexColor("#334155"),
                              spaceAfter=2*mm))
    styles.add(ParagraphStyle(name="TissueLabel",
                              fontName="Helvetica-Bold", fontSize=10,
                              alignment=TA_CENTER, spaceAfter=1*mm))
    styles.add(ParagraphStyle(name="AreaValue",
                              fontName="Helvetica", fontSize=9,
                              alignment=TA_CENTER, spaceAfter=1*mm))
    styles.add(ParagraphStyle(name="RefNote",
                              fontName="Helvetica-Oblique", fontSize=7,
                              textColor=HexColor("#94A3B8"),
                              alignment=TA_CENTER))

    elements = []

    # Header
    elements.append(Paragraph("Body Composition Analysis", styles["Title2"]))
    elements.append(Paragraph("BodySegAI — CT Body Composition Segmentation", styles["Subtitle"]))
    elements.append(Spacer(1, 2*mm))

    # Patient info table
    seg_date = datetime.now().strftime("%Y-%m-%d")
    sex_display = {"M": "Male", "F": "Female"}.get(gender_code, patient_sex or "—")
    study_date_display = study_date if study_date else "—"
    if len(study_date_display) == 8 and study_date_display.isdigit():
        study_date_display = f"{study_date_display[:4]}-{study_date_display[4:6]}-{study_date_display[6:]}"
    thickness_display = f"{slice_thickness:.1f} mm" if slice_thickness else "—"

    info_data = [
        ["Patient Name", patient_name or "—", "Exam Date", study_date_display],
        ["Gender", sex_display, "Segmentation Date", seg_date],
        ["Slice Thickness", thickness_display, "Pixel Spacing", f"{pixel_spacing[0]:.2f} x {pixel_spacing[1]:.2f} mm"],
    ]
    info_table = Table(info_data, colWidths=[35*mm, 50*mm, 40*mm, 50*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#334155")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3*mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1*mm),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, HexColor("#E2E8F0")),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 4*mm))

    # Section: Individual tissue overlays (2x2 grid)
    elements.append(Paragraph("Tissue Segmentation", styles["SectionHead"]))

    display_hu = hu_to_display(hu_image)
    tissue_images = []
    tissue_info = []

    for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
        overlay = create_single_tissue_overlay(display_hu, mask, label)
        rl_img = _np_to_rl_image(overlay, width_cm=7.5)

        color_hex = TISSUE_COLORS_HEX[label]
        name = TISSUE_NAMES[label]
        area = areas.get(label, 0)
        hu_val = mean_hu.get(label)
        hu_str = f"Mean HU: {hu_val}" if hu_val is not None else "Mean HU: —"

        tissue_images.append(rl_img)
        tissue_info.append((name, area, hu_str, color_hex))

    # Build 2x2 table of images
    img_table_data = [
        [tissue_images[0], tissue_images[1]],
        [
            _tissue_caption(tissue_info[0], styles),
            _tissue_caption(tissue_info[1], styles),
        ],
        [tissue_images[2], tissue_images[3]],
        [
            _tissue_caption(tissue_info[2], styles),
            _tissue_caption(tissue_info[3], styles),
        ],
    ]
    img_table = Table(img_table_data, colWidths=[88*mm, 88*mm])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1*mm),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 4*mm))

    # Summary table
    elements.append(Paragraph("Results Summary", styles["SectionHead"]))
    summary_data = [["Tissue", "Area (cm²)", "Mean HU"]]
    for label in [LABEL_MUSCLE, LABEL_IMAT, LABEL_SAT, LABEL_VAT]:
        name = TISSUE_NAMES[label]
        area = areas.get(label, 0)
        hu_val = mean_hu.get(label)
        hu_str = f"{hu_val}" if hu_val is not None else "—"
        summary_data.append([name, f"{area:.2f}", hu_str])

    sum_table = Table(summary_data, colWidths=[80*mm, 40*mm, 40*mm])
    sum_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#334155")),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F1F5F9")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, HexColor("#CBD5E1")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(sum_table)
    elements.append(Spacer(1, 4*mm))

    # Reference values
    if gender_code in REFERENCES:
        elements.append(Paragraph("Reference Values", styles["SectionHead"]))
        refs = REFERENCES[gender_code]

        for label, ref in refs.items():
            area = areas.get(label, 0)
            color = HexColor("#10B981")  # green

            if "cutoff_area" in ref and area > ref["cutoff_area"]:
                color = HexColor("#EF4444")  # red

            name = TISSUE_NAMES.get(label, "")
            msg = ""
            if "cutoff_area" in ref:
                cutoff = ref["cutoff_area"]
                status = "ELEVATED" if area > cutoff else "Normal"
                msg = f"{name}: {area:.1f} cm² — {status} (cutoff: {cutoff} cm²)"
            else:
                msg = f"{name}: {area:.1f} cm²"

            note = ref.get("note", "")
            source = ref.get("source", "")

            p = Paragraph(f'<font color="{color}">{msg}</font>', styles["BodySmall"])
            elements.append(p)
            if note:
                elements.append(Paragraph(note, styles["RefNote"]))
            if source:
                elements.append(Paragraph(f"Ref: {source}", styles["RefNote"]))
            elements.append(Spacer(1, 2*mm))

    # Footer
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph(
        "Generated by BodySegAI. For research purposes only. "
        "Not intended for clinical diagnosis.",
        ParagraphStyle(name="Footer", fontName="Helvetica-Oblique", fontSize=7,
                       textColor=HexColor("#94A3B8"), alignment=TA_CENTER)
    ))

    doc.build(elements)
    return buf.getvalue()


def _tissue_caption(info, styles):
    """Create a caption paragraph for a tissue overlay image."""
    name, area, hu_str, color_hex = info
    return Paragraph(
        f'<font color="{color_hex}"><b>{name}</b></font><br/>'
        f'Area: {area:.2f} cm² | {hu_str}',
        ParagraphStyle(name="cap", fontName="Helvetica", fontSize=8,
                       textColor=HexColor("#475569"), alignment=TA_CENTER,
                       leading=11)
    )
