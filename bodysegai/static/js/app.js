/* ========================================
   BodySegAI — Frontend Logic
   ======================================== */

const TISSUE_LABELS = {
    "1": { name: "Skeletal Muscle", short: "SM", color: "#EF4444" },
    "2": { name: "Inter- & Intramuscular Adipose Tissue", short: "IMAT", color: "#39FF14" },
    "5": { name: "Visceral Adipose Tissue", short: "VAT", color: "#FACC15" },
    "7": { name: "Subcutaneous Adipose Tissue", short: "SAT", color: "#25F5FC" },
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
        $id("fileName").value = data.filename || "";
        $id("studyDate").value = formatDate(data.study_date) || "\u2014";
        $id("sliceThickness").value = data.slice_thickness
            ? data.slice_thickness.toFixed(1) + " mm"
            : "\u2014";
        $id("imageMeta").textContent =
            `${data.filename} \u2014 ${data.n_slices} slice(s) \u2014 Pixel: ${data.pixel_spacing[0].toFixed(2)}\u00d7${data.pixel_spacing[1].toFixed(2)} mm`;

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
    const fname = $id("fileName").value || "\u2014";
    const sex = $id("patientSex").value === "M" ? "Male" : $id("patientSex").value === "F" ? "Female" : "";
    $id("resultsPatientInfo").textContent = `${fname}${sex ? " \u00b7 " + sex : ""}`;

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
                <div class="tissue-card-area">${area.toFixed(2)} <span>cm\u00b2</span></div>
                <div class="tissue-card-hu">Mean HU: ${hu !== null && hu !== undefined ? hu : "\u2014"}</div>
            </div>
        `;
        grid.appendChild(card);
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
        const st = slice.slice_thickness != null ? slice.slice_thickness.toFixed(1) : "\u2014";
        tr.innerHTML = `
            <td>${slice.filename || slice.slice_index + 1}</td>
            <td>${(a["1"] || 0).toFixed(2)}</td>
            <td>${(a["2"] || 0).toFixed(2)}</td>
            <td>${(a["5"] || 0).toFixed(2)}</td>
            <td>${(a["7"] || 0).toFixed(2)}</td>
            <td>${h["1"] != null ? h["1"] : "\u2014"}</td>
            <td>${h["2"] != null ? h["2"] : "\u2014"}</td>
            <td>${h["5"] != null ? h["5"] : "\u2014"}</td>
            <td>${h["7"] != null ? h["7"] : "\u2014"}</td>
            <td>${st}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ---- Downloads ----
$id("btnDownloadPdf").addEventListener("click", () => {
    window.location.href = "/api/download/pdf";
});

$id("btnDownloadExcel").addEventListener("click", () => {
    window.location.href = "/api/download/excel";
});

$id("btnDownloadExcelMulti").addEventListener("click", () => {
    window.location.href = "/api/download/excel";
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

// ---- Shutdown ----
const SHUTDOWN_MSG = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#64748B;"><h2>BodySegAI has been shut down. You can close this tab.</h2></div>';

$id("btnShutdown").addEventListener("click", async () => {
    if (!confirm("Shut down BodySegAI?")) return;
    try {
        await fetch("/api/shutdown");
    } catch (e) {
        // Expected — server dies
    }
    document.body.innerHTML = SHUTDOWN_MSG;
});

// ---- Idle timeout warning ----
const idleToast = $id("idleToast");
const idleMsg = $id("idleMsg");
const btnStayAlive = $id("btnStayAlive");

btnStayAlive.addEventListener("click", async () => {
    try { await fetch("/api/keepalive"); } catch (e) { /* ignore */ }
    idleToast.classList.remove("visible");
});

setInterval(async () => {
    try {
        const resp = await fetch("/api/idle-status");
        const data = await resp.json();
        const remaining = data.remaining_seconds;
        if (remaining <= 0) {
            document.body.innerHTML = SHUTDOWN_MSG;
            return;
        }
        if (remaining <= 300) {
            const mins = Math.ceil(remaining / 60);
            idleMsg.textContent = `Server will shut down in ${mins} min due to inactivity.`;
            idleToast.classList.add("visible");
        } else {
            idleToast.classList.remove("visible");
        }
    } catch (e) {
        // Server already gone
        document.body.innerHTML = SHUTDOWN_MSG;
    }
}, 30000);

// ---- Image zoom modal ----
const imageModal = $id("imageModal");
const imageModalImg = $id("imageModalImg");

function openModal(src) {
    imageModalImg.src = src;
    imageModal.classList.add("visible");
}

function closeModal() {
    imageModal.classList.remove("visible");
    imageModalImg.src = "";
}

$id("imageModalClose").addEventListener("click", closeModal);
imageModal.addEventListener("click", closeModal);

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});

document.addEventListener("click", (e) => {
    const img = e.target.closest(".image-frame img, .comparison-view img");
    if (img) {
        e.preventDefault();
        openModal(img.src);
    }
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
