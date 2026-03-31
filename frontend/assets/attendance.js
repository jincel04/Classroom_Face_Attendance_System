const ADMIN_TOKEN_KEY = "smart_attendance_admin_token_v1";

const state = {
  students: [],
  detections: [],
  boxes: [],
  running: false,
  elapsed: 0,
  backendAvailable: false,
  sessionId: "",
  lastEventId: 0,
  processedFrames: 0,
  totalDetections: 0,
  backendError: "",
  statusPollRef: null,
  framePollRef: null,
  healthRef: null,
  frameUrl: "",
  adminToken: loadAdminToken(),
};

const el = {
  sessionPill: document.getElementById("sessionPill"),
  backendPill: document.getElementById("backendPill"),
  feedPill: document.getElementById("feedPill"),
  rosterPill: document.getElementById("rosterPill"),
  metricTime: document.getElementById("metricTime"),
  metricDetected: document.getElementById("metricDetected"),
  metricPending: document.getElementById("metricPending"),
  metricRoster: document.getElementById("metricRoster"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearBtn: document.getElementById("clearBtn"),
  syncBtn: document.getElementById("syncBtn"),
  sessionStatus: document.getElementById("sessionStatus"),
  preview: document.getElementById("preview"),
  overlayLayer: document.getElementById("overlayLayer"),
  cameraNote: document.getElementById("cameraNote"),
  rosterList: document.getElementById("rosterList"),
  detectedList: document.getElementById("detectedList"),
};

init();

async function init() {
  bindEvents();
  renderAll();
  await checkBackend();
  if (state.backendAvailable) {
    await syncRosterFromBackend(true);
    await refreshAttendanceStatus(true);
  } else {
    setStatus("Backend unavailable. Start scripts/run_frontend_server.py", "err");
  }
  state.healthRef = setInterval(periodicRefresh, 15000);
}

function bindEvents() {
  el.startBtn.addEventListener("click", startSession);
  el.stopBtn.addEventListener("click", stopSession);
  el.clearBtn.addEventListener("click", clearSession);
  el.syncBtn.addEventListener("click", () => syncRosterFromBackend(false));
  window.addEventListener("beforeunload", cleanupView);
}

function cleanupView() {
  stopPolling();
  if (state.frameUrl) {
    URL.revokeObjectURL(state.frameUrl);
    state.frameUrl = "";
  }
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

async function periodicRefresh() {
  await checkBackend();
  if (!state.backendAvailable) {
    return;
  }
  await syncRosterFromBackend(false);
  if (!state.running) {
    await refreshAttendanceStatus(true);
  }
}

async function checkBackend() {
  try {
    const resp = await apiJson("/api/health", { cache: "no-store" });
    if (!resp.ok || !resp.data.ok) {
      throw new Error("health failed");
    }
    state.backendAvailable = true;
    setPill(el.backendPill, "Backend On", "ok");
  } catch (error) {
    state.backendAvailable = false;
    setPill(el.backendPill, "Backend Off", "err");
  }
  el.syncBtn.disabled = !state.backendAvailable;
  el.startBtn.disabled = !state.backendAvailable || state.running;
  el.stopBtn.disabled = !state.running;
}

async function syncRosterFromBackend(showStatus = true) {
  if (!state.backendAvailable) {
    if (showStatus) {
      setStatus("Backend not available.", "err");
    }
    return;
  }

  const resp = await apiJson("/api/students", { cache: "no-store" });
  if (!resp.ok || !resp.data.ok) {
    if (showStatus) {
      setStatus(resp.data.error || "Failed to sync roster from backend.", "err");
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

  renderAll();
  if (showStatus) {
    setStatus("Roster synced from backend.", "ok");
  }
}

async function refreshAttendanceStatus(silent = false) {
  if (!state.backendAvailable) {
    return;
  }
  const resp = await apiJson("/api/attendance/status", { cache: "no-store" });
  if (!resp.ok || !resp.data.ok) {
    if (!silent) {
      setStatus(resp.data.error || "Failed to fetch attendance status.", "err");
    }
    return;
  }
  applyAttendanceStatus(resp.data.attendance || {});
  renderAll();
}

function applyAttendanceStatus(status) {
  state.running = Boolean(status.running);
  state.sessionId = String(status.sessionId || "");
  state.elapsed = Number(status.elapsedSec || 0);
  state.processedFrames = Number(status.processedFrames || 0);
  state.totalDetections = Number(status.detections || 0);
  state.backendError = String(status.error || "");
  if (status.lastEventId != null) {
    state.lastEventId = Math.max(state.lastEventId, Number(status.lastEventId) || 0);
  }
  el.startBtn.disabled = !state.backendAvailable || state.running;
  el.stopBtn.disabled = !state.running;
}

async function startSession() {
  if (state.running) {
    return;
  }
  if (!state.backendAvailable) {
    setStatus("Backend unavailable.", "err");
    return;
  }
  if (!state.students.length) {
    setStatus("No students enrolled in backend. Open Enrollment page first.", "err");
    return;
  }

  setStatus("Starting backend attendance session...", "info");
  const resp = await apiJson(
    "/api/attendance/start",
    {
      method: "POST",
      body: JSON.stringify({ cameraIndex: 0 }),
    },
    true
  );
  if (!resp.ok || !resp.data.ok) {
    setStatus(resp.data.error || "Failed to start attendance.", "err");
    return;
  }

  state.detections = [];
  state.boxes = [];
  state.lastEventId = 0;
  applyAttendanceStatus(resp.data.attendance || {});
  beginPolling();
  await pollAttendance();
  setStatus("Session started. Real backend detections are live.", "ok");
  renderAll();
}

async function stopSession() {
  if (!state.running) {
    return;
  }
  const resp = await apiJson("/api/attendance/stop", { method: "POST" }, true);
  if (!resp.ok || !resp.data.ok) {
    setStatus(resp.data.error || "Failed to stop attendance session.", "err");
    return;
  }

  applyAttendanceStatus(resp.data.attendance || {});
  state.boxes = [];
  stopPolling();
  renderAll();
  setStatus("Session stopped.", "info");
}

function clearSession() {
  if (state.running) {
    stopSession();
  }
  state.elapsed = 0;
  state.detections = [];
  state.boxes = [];
  state.lastEventId = 0;
  setStatus("Session data cleared.", "info");
  renderAll();
}

function beginPolling() {
  if (state.statusPollRef === null) {
    state.statusPollRef = setInterval(() => {
      pollAttendance();
    }, 900);
  }
  if (state.framePollRef === null) {
    state.framePollRef = setInterval(() => {
      pollFrame();
    }, 350);
  }
}

function stopPolling() {
  if (state.statusPollRef !== null) {
    clearInterval(state.statusPollRef);
    state.statusPollRef = null;
  }
  if (state.framePollRef !== null) {
    clearInterval(state.framePollRef);
    state.framePollRef = null;
  }
  el.cameraNote.textContent = state.running ? "Backend stream waiting" : "Camera idle";
}

async function pollAttendance() {
  if (!state.backendAvailable) {
    return;
  }

  const resp = await apiJson(
    "/api/attendance/events?since=" + encodeURIComponent(String(state.lastEventId)),
    { cache: "no-store" }
  );
  if (!resp.ok || !resp.data.ok) {
    setStatus(resp.data.error || "Failed to poll attendance events.", "err");
    return;
  }

  const events = Array.isArray(resp.data.events) ? resp.data.events : [];
  if (events.length) {
    for (const event of events) {
      const eventId = Number(event.eventId || 0);
      if (eventId > state.lastEventId) {
        state.lastEventId = eventId;
      }
      state.detections.unshift({
        eventId,
        studentId: String(event.studentId || ""),
        name: String(event.name || ""),
        confidence: Number(event.confidence || 0),
        detectedAt: String(event.detectedAt || ""),
      });
    }
    state.detections = state.detections
      .filter((entry) => entry.studentId && entry.name)
      .slice(0, 300);
  }

  state.boxes = Array.isArray(resp.data.boxes) ? resp.data.boxes.slice(0, 24) : [];
  applyAttendanceStatus(resp.data.status || {});
  renderAll();

  if (!state.running) {
    stopPolling();
    if (state.backendError) {
      setStatus("Session ended with error: " + state.backendError, "err");
    }
  }
}

async function pollFrame() {
  if (!state.running) {
    return;
  }
  try {
    const resp = await fetch("/api/attendance/frame?t=" + Date.now(), { cache: "no-store" });
    if (!resp.ok) {
      el.cameraNote.textContent = "Backend stream waiting";
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    if (state.frameUrl) {
      URL.revokeObjectURL(state.frameUrl);
    }
    state.frameUrl = url;
    el.preview.src = url;
    el.cameraNote.textContent = "Backend camera connected";
  } catch (error) {
    el.cameraNote.textContent = "Backend stream error";
  }
}

function renderAll() {
  renderSummary();
  renderControls();
  renderRoster();
  renderDetections();
  renderBoxes();
}

function renderSummary() {
  const detectedIds = new Set(state.detections.map((entry) => entry.studentId));
  const detected = detectedIds.size;
  const pending = Math.max(0, state.students.length - detected);
  el.metricTime.textContent = formatDuration(Math.max(0, Math.floor(state.elapsed)));
  el.metricDetected.textContent = String(detected);
  el.metricPending.textContent = String(pending);
  el.metricRoster.textContent = String(state.students.length);
  el.rosterPill.textContent = state.students.length + " Students";
  setPill(el.sessionPill, state.running ? "Session On" : "Session Off", state.running ? "ok" : "warn");
  const feedLive = state.running || state.detections.length > 0;
  setPill(el.feedPill, feedLive ? "Live Feed" : "Waiting", feedLive ? "ok" : "warn");
}

function renderControls() {
  el.startBtn.disabled = !state.backendAvailable || state.running;
  el.stopBtn.disabled = !state.running;
}

function renderRoster() {
  if (!state.students.length) {
    el.rosterList.innerHTML =
      "<div class='empty'>No students in backend roster. Open Enrollment page to add students.</div>";
    return;
  }

  const detectedIds = new Set(state.detections.map((entry) => entry.studentId));
  el.rosterList.innerHTML = state.students
    .map((student) => {
      const hit = detectedIds.has(student.studentId);
      const meta = [
        "Enrollment No: " + student.studentId,
        student.section ? "Section: " + student.section : "",
        student.rollNo ? "Roll: " + student.rollNo : "",
      ]
        .filter(Boolean)
        .join(" | ");

      return `
          <div class="item">
            <div>
              <strong>${escapeHtml(student.name)}</strong>
              <small>${escapeHtml(meta)}</small>
            </div>
            <span class="chip ${hit ? "ok" : "wait"}">${hit ? "Detected" : "Pending"}</span>
          </div>
        `;
    })
    .join("");
}

function renderDetections() {
  if (!state.detections.length) {
    el.detectedList.innerHTML = "<div class='empty'>No detections yet. Start session to begin.</div>";
    return;
  }

  el.detectedList.innerHTML = state.detections
    .map((entry) => {
      return `
          <div class="item">
            <div>
              <strong>${escapeHtml(entry.name)}</strong>
              <small>Enrollment No ${escapeHtml(entry.studentId)} | ${escapeHtml(
                formatTime(entry.detectedAt)
              )}</small>
            </div>
            <span class="chip ok">${Math.round(entry.confidence * 100)}%</span>
          </div>
        `;
    })
    .join("");
}

function renderBoxes() {
  if (!state.running || !state.boxes.length) {
    el.overlayLayer.innerHTML = "";
    return;
  }
  el.overlayLayer.innerHTML = state.boxes
    .map((box) => {
      const color =
        box.tag === "unknown" ? "#ef4444" : box.tag === "marked" ? "#22c55e" : "#2dd4bf";
      const left = Number(box.left || 0);
      const top = Number(box.top || 0);
      const width = Number(box.width || 0);
      const height = Number(box.height || 0);
      return `
          <div class="face-box" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%;border-color:${color};">
            <label>${escapeHtml(String(box.label || ""))}</label>
          </div>
        `;
    })
    .join("");
}

function setStatus(text, type) {
  el.sessionStatus.textContent = text;
  el.sessionStatus.className = "status";
  if (type) {
    el.sessionStatus.classList.add(type);
  }
}

function setPill(node, text, type) {
  node.textContent = text;
  node.className = "pill";
  if (type) {
    node.classList.add(type);
  }
}

function formatDuration(totalSeconds) {
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const ss = String(totalSeconds % 60).padStart(2, "0");
  return mm + ":" + ss;
}

function formatTime(iso) {
  if (!iso) {
    return "--:--:--";
  }
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}
