/* ========================================
   BodySegAI — Frontend Logic
   ======================================== */

const TISSUE_LABELS = {
    "1": { name: "Skeletal Muscle", short: "SM", color: "#EF4444" },
    "2": { name: "IMAT", short: "IMAT", color: "#F59E0B" },
    "5": { name: "SAT", short: "SAT", color: "#3B82F6" },
    "7": { name: "VAT", short: "VAT", color: "#10B981" },
};

let currentMode = null;

// ---- DOM refs ----
const $id = (id) => document.getElementById(id);
const dropZone = $id("dropZone");
const fileInput = $id("fileInput");
const folderInput = $id("folderInput");
const loadingOverlay = $id("loadingOverlay");
const loadingText = $id("loadingText");

// ---- Steps ----
function showStep(stepId) {
    document.querySelectorAll(".step").forEach((s) => s.classList.remove("active"));
    const el = $id(stepId);
    if (el) el.classList.add("active");
}

// ---- Upload / Drag & Drop ----
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
});

folderInput.addEventListener("change", () => {
    if (folderInput.files.length) handleFiles(folderInput.files);
});

async function handleFiles(fileList) {
    showLoading("Loading files...");

    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("files", fileList[i]);
    }

    try {
        const resp = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await resp.json();

        if (data.error) {
            alert("Error: " + data.error);
            hideLoading();
            return;
        }

        currentMode = data.mode;

        // Set preview
        $id("previewImage").src = "data:image/png;base64," + data.preview;

        // Set patient info
        $id("patientName").value = data.patient_name || "";
        $id("studyDate").value = formatDate(data.study_date) || "—";
        $id("sliceThickness").value = data.slice_thickness
            ? data.slice_thickness.toFixed(1) + " mm"
            : "—";
        $id("imageMeta").textContent =
            `${data.filename} — ${data.n_slices} slice(s) — Pixel: ${data.pixel_spacing[0].toFixed(2)}×${data.pixel_spacing[1].toFixed(2)} mm`;

        // Set gender
        const sexSelect = $id("patientSex");
        if (data.patient_sex) {
            const g = data.patient_sex[0].toUpperCase();
            sexSelect.value = g === "M" ? "M" : g === "F" ? "F" : "";
        } else {
            sexSelect.value = "";
        }

        if (currentMode === "single") {
            $id("patientInfoCard").style.display = "block";
            $id("multiInfoBar").style.display = "none";
        } else {
            $id("patientInfoCard").style.display = "none";
            $id("multiInfoBar").style.display = "flex";
            $id("multiSliceCount").textContent = data.n_slices;
        }

        hideLoading();
        showStep("stepPreview");
    } catch (err) {
        alert("Upload failed: " + err.message);
        hideLoading();
    }
}

// ---- Segmentation ----
$id("btnSegment").addEventListener("click", runSegmentation);
$id("btnSegmentMulti").addEventListener("click", runSegmentation);

async function runSegmentation() {
    showLoading("Running segmentation...");

    const body = {
        patient_name: $id("patientName").value,
        patient_sex: $id("patientSex").value,
    };

    try {
        const resp = await fetch("/api/segment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (data.error) {
            alert("Error: " + data.error);
            hideLoading();
            return;
        }

        hideLoading();

        if (data.mode === "single") {
            renderSingleResults(data);
            showStep("stepResultsSingle");
        } else {
            renderMultiResults(data);
            showStep("stepResultsMulti");
        }
    } catch (err) {
        alert("Segmentation failed: " + err.message);
        hideLoading();
    }
}

// ---- Render Single Results ----
function renderSingleResults(data) {
    const slice = data.slices[0];

    // Patient info line
    const name = $id("patientName").value || "—";
    const sex = $id("patientSex").value === "M" ? "Male" : $id("patientSex").value === "F" ? "Female" : "";
    $id("resultsPatientInfo").textContent = `${name}${sex ? " · " + sex : ""}`;

    // Raw + overlay
    $id("resultRaw").src = "data:image/png;base64," + slice.raw_preview;
    $id("resultOverlay").src = "data:image/png;base64," + slice.overlay_preview;

    // Tissue cards
    const grid = $id("tissueGrid");
    grid.innerHTML = "";

    for (const [label, info] of Object.entries(TISSUE_LABELS)) {
        const area = slice.areas[label] || 0;
        const hu = slice.mean_hu[label];
        const overlayImg = slice.tissue_overlays ? slice.tissue_overlays[label] : null;

        const card = document.createElement("div");
        card.className = "tissue-card";
        card.innerHTML = `
            ${overlayImg ? `<div class="image-frame"><img src="data:image/png;base64,${overlayImg}" alt="${info.name}"></div>` : ""}
            <div class="tissue-card-info">
                <div class="tissue-card-name" style="color:${info.color}">${info.name}</div>
                <div class="tissue-card-area">${area.toFixed(2)} <span>cm²</span></div>
                <div class="tissue-card-hu">Mean HU: ${hu !== null && hu !== undefined ? hu : "—"}</div>
            </div>
        `;
        grid.appendChild(card);
    }

    // Reference values
    const refSection = $id("referenceSection");
    const refContent = $id("referenceContent");
    if (data.references && Object.keys(data.references).length > 0) {
        refContent.innerHTML = "";
        for (const [label, ref] of Object.entries(data.references)) {
            const div = document.createElement("div");
            div.className = "ref-item";
            div.innerHTML = `
                <div class="ref-status ${ref.status}"></div>
                <div class="ref-message">${ref.message}</div>
                <div class="ref-source">${ref.source}</div>
            `;
            refContent.appendChild(div);

            if (ref.note) {
                const note = document.createElement("div");
                note.className = "ref-item";
                note.innerHTML = `<div class="ref-status info"></div><div class="ref-message" style="font-size:12px;color:var(--text-muted)">${ref.note}</div>`;
                refContent.appendChild(note);
            }
        }
        refSection.style.display = "block";
    } else {
        refSection.style.display = "none";
    }
}

// ---- Render Multi Results ----
function renderMultiResults(data) {
    $id("multiResultsSummary").textContent =
        `${data.n_slices} slices processed successfully`;

    const tbody = $id("multiResultsBody");
    tbody.innerHTML = "";

    for (const slice of data.slices) {
        const tr = document.createElement("tr");
        const a = slice.areas;
        const h = slice.mean_hu;
        tr.innerHTML = `
            <td>${slice.slice_index + 1}</td>
            <td>${(a["1"] || 0).toFixed(2)}</td>
            <td>${(a["2"] || 0).toFixed(2)}</td>
            <td>${(a["5"] || 0).toFixed(2)}</td>
            <td>${(a["7"] || 0).toFixed(2)}</td>
            <td>${h["1"] != null ? h["1"] : "—"}</td>
            <td>${h["2"] != null ? h["2"] : "—"}</td>
            <td>${h["5"] != null ? h["5"] : "—"}</td>
            <td>${h["7"] != null ? h["7"] : "—"}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ---- Downloads ----
$id("btnDownloadPdf").addEventListener("click", () => {
    window.location.href = "/api/download/pdf";
});

$id("btnDownloadCsv").addEventListener("click", () => {
    window.location.href = "/api/download/csv";
});

$id("btnDownloadCsvMulti").addEventListener("click", () => {
    window.location.href = "/api/download/csv";
});

$id("btnDownloadMask").addEventListener("click", () => {
    window.location.href = "/api/download/mask";
});

$id("btnDownloadMaskMulti").addEventListener("click", () => {
    window.location.href = "/api/download/mask";
});

// ---- Load Corrected Mask ----
$id("correctedMaskInput").addEventListener("change", async function () {
    if (!this.files.length) return;

    showLoading("Loading corrected mask...");

    const formData = new FormData();
    formData.append("mask", this.files[0]);

    try {
        const resp = await fetch("/api/upload_mask", { method: "POST", body: formData });
        const data = await resp.json();

        if (data.error) {
            alert("Error: " + data.error);
            hideLoading();
            return;
        }

        hideLoading();

        if (data.mode === "single") {
            renderSingleResults(data);
        } else {
            renderMultiResults(data);
        }
    } catch (err) {
        alert("Failed to load mask: " + err.message);
        hideLoading();
    }

    this.value = "";
});

// ---- Start Over ----
[$id("btnBack"), $id("btnBackMulti"), $id("btnNewAnalysis"), $id("btnNewAnalysisMulti")].forEach(
    (btn) => {
        if (btn)
            btn.addEventListener("click", () => {
                currentMode = null;
                fileInput.value = "";
                folderInput.value = "";
                showStep("stepUpload");
            });
    }
);

// ---- Helpers ----
function showLoading(text) {
    loadingText.textContent = text || "Processing...";
    loadingOverlay.style.display = "flex";
}

function hideLoading() {
    loadingOverlay.style.display = "none";
}

function formatDate(dateStr) {
    if (!dateStr) return "";
    if (dateStr.length === 8 && /^\d+$/.test(dateStr)) {
        return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6)}`;
    }
    return dateStr;
}
