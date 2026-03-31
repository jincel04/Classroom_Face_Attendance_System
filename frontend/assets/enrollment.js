const ADMIN_TOKEN_KEY = "smart_attendance_admin_token_v1";

const state = {
  students: [],
  filtered: [],
  editingId: null,
  search: "",
  backendAvailable: false,
  busy: false,
  adminToken: loadAdminToken(),
};

const el = {
  backendPill: document.getElementById("backendPill"),
  rosterPill: document.getElementById("rosterPill"),
  metricTotal: document.getElementById("metricTotal"),
  metricBusy: document.getElementById("metricBusy"),
  metricMode: document.getElementById("metricMode"),
  metricSearch: document.getElementById("metricSearch"),
  enrollForm: document.getElementById("enrollForm"),
  enrollmentNo: document.getElementById("enrollmentNo"),
  fullName: document.getElementById("fullName"),
  sectionName: document.getElementById("sectionName"),
  rollNo: document.getElementById("rollNo"),
  saveBtn: document.getElementById("saveBtn"),
  cameraBtn: document.getElementById("cameraBtn"),
  cancelEditBtn: document.getElementById("cancelEditBtn"),
  numImages: document.getElementById("numImages"),
  cameraIndex: document.getElementById("cameraIndex"),
  captureInterval: document.getElementById("captureInterval"),
  minProbability: document.getElementById("minProbability"),
  cameraWidth: document.getElementById("cameraWidth"),
  cameraHeight: document.getElementById("cameraHeight"),
  bulkInput: document.getElementById("bulkInput"),
  bulkImportBtn: document.getElementById("bulkImportBtn"),
  bulkClearBtn: document.getElementById("bulkClearBtn"),
  searchInput: document.getElementById("searchInput"),
  syncBtn: document.getElementById("syncBtn"),
  exportBtn: document.getElementById("exportBtn"),
  clearAllBtn: document.getElementById("clearAllBtn"),
  rosterBody: document.getElementById("rosterBody"),
  formStatus: document.getElementById("formStatus"),
  cameraStatus: document.getElementById("cameraStatus"),
  bulkStatus: document.getElementById("bulkStatus"),
  rosterStatus: document.getElementById("rosterStatus"),
};

init();

async function init() {
  bindEvents();
  applySearch();
  renderAll();
  await checkBackend();
  if (state.backendAvailable) {
    await syncFromBackend(true);
  } else {
    setStatus(el.rosterStatus, "Backend unavailable. Start scripts/run_frontend_server.py", "err");
  }
  setInterval(periodicRefresh, 15000);
}

function bindEvents() {
  el.enrollForm.addEventListener("submit", onSaveDetails);
  el.cameraBtn.addEventListener("click", onCameraEnrollment);
  el.cancelEditBtn.addEventListener("click", cancelEdit);
  el.bulkImportBtn.addEventListener("click", onBulkImport);
  el.bulkClearBtn.addEventListener("click", () => {
    el.bulkInput.value = "";
    setStatus(el.bulkStatus, "", "");
  });
  el.searchInput.addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    applySearch();
    renderAll();
  });
  el.syncBtn.addEventListener("click", () => syncFromBackend(false));
  el.exportBtn.addEventListener("click", exportCsv);
  el.clearAllBtn.addEventListener("click", clearAllStudents);
}

async function periodicRefresh() {
  await checkBackend();
  if (!state.backendAvailable || state.busy) {
    return;
  }
  await syncFromBackend(false);
}

function loadAdminToken() {
  try {
    return (localStorage.getItem(ADMIN_TOKEN_KEY) || "").trim();
  } catch (error) {
    return "";
  }
}

function saveAdminToken(token) {
  try {
    localStorage.setItem(ADMIN_TOKEN_KEY, token || "");
  } catch (error) {
    // Ignore storage failures.
  }
}

async function apiJson(path, options = {}, requireAdmin = false) {
  const requestOptions = { ...options };
  const baseHeaders = new Headers(requestOptions.headers || {});
  if (requestOptions.body && !baseHeaders.has("Content-Type")) {
    baseHeaders.set("Content-Type", "application/json");
  }

  let promptedForToken = false;
  while (true) {
    const headers = new Headers(baseHeaders);
    if (requireAdmin && state.adminToken) {
      headers.set("X-Admin-Token", state.adminToken);
    }

    const resp = await fetch(path, { ...requestOptions, headers });
    const contentType = String(resp.headers.get("content-type") || "");
    let data = {};
    if (contentType.includes("application/json")) {
      try {
        data = await resp.json();
      } catch (error) {
        data = {};
      }
    }

    if (resp.status === 401 && requireAdmin && !promptedForToken) {
      const token = window.prompt("Enter admin token:");
      if (!token || !token.trim()) {
        return {
          ok: false,
          status: resp.status,
          data: data && data.error ? data : { ok: false, error: "Admin token required." },
        };
      }
      state.adminToken = token.trim();
      saveAdminToken(state.adminToken);
      promptedForToken = true;
      continue;
    }

    if (resp.status === 401 && requireAdmin && promptedForToken) {
      state.adminToken = "";
      saveAdminToken("");
    }

    return { ok: resp.ok, status: resp.status, data };
  }
}

async function checkBackend() {
  try {
    const resp = await apiJson("/api/health");
    if (!resp.ok || !resp.data.ok) {
      throw new Error("health check failed");
    }
    state.backendAvailable = true;
    setPill(el.backendPill, "Backend On", "ok");
  } catch (error) {
    state.backendAvailable = false;
    setPill(el.backendPill, "Backend Off", "err");
  }
  applyBusyState();
}

async function syncFromBackend(showStatus = true) {
  if (!state.backendAvailable) {
    if (showStatus) {
      setStatus(el.rosterStatus, "Backend not available.", "err");
    }
    return;
  }
  const resp = await apiJson("/api/students", { method: "GET" });
  if (!resp.ok || !resp.data.ok) {
    if (showStatus) {
      setStatus(el.rosterStatus, "Failed to fetch students from backend.", "err");
    }
    return;
  }

  const rows = Array.isArray(resp.data.students) ? resp.data.students : [];
  state.students = rows
    .map((row) => ({
      studentId: String(row.enrollmentNo || row.studentId || "").trim(),
      name: String(row.name || "").trim(),
      section: String(row.section || "").trim(),
      rollNo: String(row.rollNo || "").trim(),
      createdAt: String(row.createdAt || ""),
    }))
    .filter((row) => row.studentId && row.name);
  applySearch();
  renderAll();
  if (showStatus) {
    setStatus(el.rosterStatus, "Roster synced from backend.", "ok");
  }
}

function applySearch() {
  if (!state.search) {
    state.filtered = [...state.students];
    return;
  }
  state.filtered = state.students.filter((row) => {
    const hay = [row.studentId, row.name, row.section || "", row.rollNo || ""]
      .join(" ")
      .toLowerCase();
    return hay.includes(state.search);
  });
}

function readForm() {
  const studentId = el.enrollmentNo.value.trim();
  const name = el.fullName.value.trim();
  const section = el.sectionName.value.trim();
  const rollNo = el.rollNo.value.trim();
  if (!studentId || !name) {
    setStatus(el.formStatus, "Enrollment No and Full Name are required.", "err");
    return null;
  }
  return { studentId, name, section, rollNo };
}

async function onSaveDetails(event) {
  event.preventDefault();
  if (state.busy) {
    return;
  }
  if (!state.backendAvailable) {
    setStatus(el.formStatus, "Backend not available.", "err");
    return;
  }
  const payload = readForm();
  if (!payload) {
    return;
  }

  setBusy(true);
  try {
    const resp = await apiJson(
      "/api/students",
      {
        method: "POST",
        body: JSON.stringify({
          enrollmentNo: payload.studentId,
          oldStudentId: state.editingId || payload.studentId,
          name: payload.name,
          section: payload.section,
          rollNo: payload.rollNo,
        }),
      },
      true
    );
    if (!resp.ok || !resp.data.ok) {
      setStatus(el.formStatus, resp.data.error || "Failed to save student.", "err");
      return;
    }

    await syncFromBackend(false);
    if (!state.editingId) {
      resetForm();
      setStatus(el.formStatus, "Student saved to backend.", "ok");
    } else {
      cancelEdit();
      setStatus(el.formStatus, "Student updated in backend.", "ok");
    }
  } finally {
    setBusy(false);
  }
}

async function onCameraEnrollment() {
  if (state.busy) {
    return;
  }
  if (!state.backendAvailable) {
    setStatus(el.cameraStatus, "Backend not connected. Run scripts/run_frontend_server.py", "err");
    return;
  }
  const payload = readForm();
  if (!payload) {
    return;
  }

  const numImages = Number(el.numImages.value || 25);
  const cameraIndex = Number(el.cameraIndex.value || 0);
  const captureInterval = Number(el.captureInterval.value || 0.25);
  const minProbability = Number(el.minProbability.value || 0.9);
  const cameraWidth = Number(el.cameraWidth.value || 640);
  const cameraHeight = Number(el.cameraHeight.value || 480);

  if (!Number.isInteger(numImages) || numImages < 1 || numImages > 300) {
    setStatus(el.cameraStatus, "Images must be between 1 and 300.", "err");
    return;
  }
  if (!Number.isInteger(cameraIndex) || cameraIndex < 0) {
    setStatus(el.cameraStatus, "Camera index must be 0 or higher.", "err");
    return;
  }
  if (!(captureInterval > 0)) {
    setStatus(el.cameraStatus, "Capture interval must be greater than 0.", "err");
    return;
  }

  setBusy(true);
  setStatus(
    el.cameraStatus,
    "Starting camera enrollment. OpenCV window will open. Press Q to stop early.",
    "info"
  );

  try {
    const resp = await apiJson(
      "/api/enroll/webcam",
      {
        method: "POST",
        body: JSON.stringify({
          enrollmentNo: payload.studentId,
          oldStudentId: state.editingId || payload.studentId,
          name: payload.name,
          section: payload.section,
          rollNo: payload.rollNo,
          numImages,
          cameraIndex,
          captureInterval,
          minProbability,
          cameraWidth,
          cameraHeight,
          buildEmbeddings: true,
        }),
      },
      true
    );

    if (!resp.ok || !resp.data.ok) {
      setStatus(el.cameraStatus, resp.data.error || "Camera enrollment failed.", "err");
      return;
    }

    await syncFromBackend(false);
    setStatus(
      el.cameraStatus,
      "Completed. Saved " +
        resp.data.saved +
        "/" +
        resp.data.requested +
        " images. Embeddings: " +
        resp.data.embeddingsBuilt,
      "ok"
    );
    setStatus(el.formStatus, "Student synced after camera enrollment.", "ok");
  } finally {
    setBusy(false);
  }
}

function startEdit(studentId) {
  const target = state.students.find((row) => row.studentId === studentId);
  if (!target) {
    return;
  }
  state.editingId = target.studentId;
  el.enrollmentNo.value = target.studentId;
  el.fullName.value = target.name;
  el.sectionName.value = target.section || "";
  el.rollNo.value = target.rollNo || "";
  el.cancelEditBtn.hidden = false;
  setStatus(el.formStatus, "Editing Enrollment No " + target.studentId, "info");
}

function cancelEdit() {
  state.editingId = null;
  el.cancelEditBtn.hidden = true;
  resetForm();
  setStatus(el.formStatus, "Edit cancelled.", "info");
  el.enrollmentNo.focus();
}

function resetForm() {
  el.enrollForm.reset();
}

async function deleteStudent(studentId) {
  if (state.busy) {
    return;
  }
  setBusy(true);
  try {
    const resp = await apiJson(
      "/api/students/" + encodeURIComponent(studentId),
      { method: "DELETE" },
      true
    );
    if (!resp.ok || !resp.data.ok) {
      setStatus(el.rosterStatus, resp.data.error || "Failed to delete student.", "err");
      return;
    }
    if (state.editingId === studentId) {
      cancelEdit();
    }
    await syncFromBackend(false);
    setStatus(el.rosterStatus, "Student deleted.", "info");
  } finally {
    setBusy(false);
  }
}

async function onBulkImport() {
  if (state.busy) {
    return;
  }
  if (!state.backendAvailable) {
    setStatus(el.bulkStatus, "Backend not available.", "err");
    return;
  }

  const text = el.bulkInput.value.trim();
  if (!text) {
    setStatus(el.bulkStatus, "Paste lines before importing.", "err");
    return;
  }

  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  let saved = 0;
  let skipped = 0;

  setBusy(true);
  try {
    for (const line of lines) {
      const cols = line.split(",").map((part) => part.trim());
      const payload = {
        studentId: cols[0] || "",
        name: cols[1] || "",
        section: cols[2] || "",
        rollNo: cols[3] || "",
      };
      if (!payload.studentId || !payload.name) {
        skipped += 1;
        continue;
      }

      const resp = await apiJson(
        "/api/students",
        {
          method: "POST",
          body: JSON.stringify({
            enrollmentNo: payload.studentId,
            name: payload.name,
            section: payload.section,
            rollNo: payload.rollNo,
          }),
        },
        true
      );
      if (resp.ok && resp.data.ok) {
        saved += 1;
      } else {
        skipped += 1;
      }
    }

    await syncFromBackend(false);
    setStatus(el.bulkStatus, "Import done. Saved: " + saved + ", Skipped: " + skipped, "ok");
  } finally {
    setBusy(false);
  }
}

function exportCsv() {
  if (!state.students.length) {
    setStatus(el.rosterStatus, "No students to export.", "err");
    return;
  }
  const header = ["enrollment_no", "name", "section", "roll_no", "created_at"];
  const rows = state.students.map((row) => [
    row.studentId,
    row.name,
    row.section || "",
    row.rollNo || "",
    row.createdAt || "",
  ]);
  const csv = [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "enrollment_roster.csv";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  setStatus(el.rosterStatus, "CSV exported.", "ok");
}

async function clearAllStudents() {
  if (state.busy) {
    return;
  }
  if (!state.students.length) {
    setStatus(el.rosterStatus, "Roster already empty.", "info");
    return;
  }
  const confirmed = window.confirm("Delete all students from backend roster?");
  if (!confirmed) {
    return;
  }

  setBusy(true);
  try {
    const resp = await apiJson("/api/students", { method: "DELETE" }, true);
    if (!resp.ok || !resp.data.ok) {
      setStatus(el.rosterStatus, resp.data.error || "Failed to clear roster.", "err");
      return;
    }
    cancelEdit();
    await syncFromBackend(false);
    setStatus(el.rosterStatus, "All students cleared.", "info");
  } finally {
    setBusy(false);
  }
}

function setBusy(value) {
  state.busy = value;
  applyBusyState();
  el.metricBusy.textContent = value ? "Yes" : "No";
}

function applyBusyState() {
  const blocked = state.busy || !state.backendAvailable;
  el.saveBtn.disabled = blocked;
  el.cameraBtn.disabled = blocked;
  el.cancelEditBtn.disabled = state.busy;
  el.bulkImportBtn.disabled = blocked;
  el.bulkClearBtn.disabled = state.busy;
  el.syncBtn.disabled = blocked;
  el.exportBtn.disabled = state.busy;
  el.clearAllBtn.disabled = blocked;
}

function renderAll() {
  renderSummary();
  renderTable();
}

function renderSummary() {
  el.metricTotal.textContent = String(state.students.length);
  el.metricMode.textContent = "Backend";
  el.metricSearch.textContent = String(state.filtered.length);
  el.rosterPill.textContent = state.students.length + " Students";
}

function renderTable() {
  if (!state.filtered.length) {
    el.rosterBody.innerHTML = "<tr><td colspan='5'>No students found.</td></tr>";
    return;
  }

  el.rosterBody.innerHTML = state.filtered
    .map((row) => {
      return `
          <tr>
            <td>${escapeHtml(row.studentId)}</td>
            <td>${escapeHtml(row.name)}</td>
            <td>${escapeHtml(row.section || "-")}</td>
            <td>${escapeHtml(row.rollNo || "-")}</td>
            <td>
              <div class="actions-inline">
                <button class="btn btn-soft btn-mini" type="button" data-edit="${escapeHtml(
                  row.studentId
                )}">Edit</button>
                <button class="btn btn-danger btn-mini" type="button" data-delete="${escapeHtml(
                  row.studentId
                )}">Delete</button>
              </div>
            </td>
          </tr>
        `;
    })
    .join("");

  el.rosterBody.querySelectorAll("[data-edit]").forEach((btn) => {
    btn.addEventListener("click", () => startEdit(btn.dataset.edit));
  });
  el.rosterBody.querySelectorAll("[data-delete]").forEach((btn) => {
    btn.addEventListener("click", () => deleteStudent(btn.dataset.delete));
  });
}

function setStatus(node, text, type) {
  node.textContent = text;
  node.className = "status";
  if (type) {
    node.classList.add(type);
  }
}

function setPill(node, text, type) {
  node.textContent = text;
  node.className = "pill";
  if (type) {
    node.classList.add(type);
  }
}

function csvCell(value) {
  const escaped = String(value).replaceAll("\"", "\"\"");
  return "\"" + escaped + "\"";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}
