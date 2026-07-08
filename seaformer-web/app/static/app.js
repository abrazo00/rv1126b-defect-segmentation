const previewLimit = window.APP_CONFIG.previewLimit || 3;
const eventPreviewLimit = window.APP_CONFIG.eventPreviewLimit || 3;
const comparePreprocessImageInput = document.getElementById("comparePreprocessImage");
const preprocessImageInput = document.getElementById("preprocessImage");
const singleImageInput = document.getElementById("singleImage");
const batchImagesInput = document.getElementById("batchImages");
const comparePreprocessPreviewEl = document.getElementById("comparePreprocessPreview");
const preprocessPreviewEl = document.getElementById("preprocessPreview");
const singlePreviewEl = document.getElementById("singlePreview");
const batchPreviewEl = document.getElementById("batchPreview");
const demoPreviewPanelEl = document.getElementById("demoPreviewPanel");
const demoLiveStripEl = document.getElementById("demoLiveStrip");
const pipelinePanelEl = document.getElementById("pipelinePanel");
const inputMatrixPanelEl = document.getElementById("inputMatrixPanel");
const performancePanelEl = document.getElementById("performancePanel");
const stabilityPanelEl = document.getElementById("stabilityPanel");
const robustnessScenariosEl = document.getElementById("robustnessScenarios");
const exportStatusEl = document.getElementById("exportStatus");
const videoStreamEl = document.getElementById("videoStream");
const videoStatusEl = document.getElementById("videoStatus");
const videoUploadStatusEl = document.getElementById("videoUploadStatus");
const videoLibraryDropdownEl = document.getElementById("videoLibraryDropdown");
const videoLibraryTriggerEl = document.getElementById("videoLibraryTrigger");
const videoLibraryMenuEl = document.getElementById("videoLibraryMenu");
const videoSourceEl = document.getElementById("videoSource");
const videoUploadInputEl = document.getElementById("videoUploadInput");
const videoPlaceholderEl = document.getElementById("videoPlaceholder");
const videoInfoPanelEl = document.getElementById("videoInfoPanel");
const mobileCameraVideoEl = document.getElementById("mobileCameraVideo");
const mobileCameraCanvasEl = document.getElementById("mobileCameraCanvas");
const mobileCameraResultEl = document.getElementById("mobileCameraResult");
const mobileCameraPlaceholderEl = document.getElementById("mobileCameraPlaceholder");
const mobileCameraResultPlaceholderEl = document.getElementById("mobileCameraResultPlaceholder");
const mobileCameraStatsEl = document.getElementById("mobileCameraStats");
const mobileCameraStatusEl = document.getElementById("mobileCameraStatus");
const serviceHealthPanelEl = document.getElementById("serviceHealthPanel");
const systemStatsPanelEl = document.getElementById("systemStatsPanel");
const eventFeedEl = document.getElementById("eventFeed");
const recentTasksEl = document.getElementById("recentTasks");
const lightboxEl = document.getElementById("lightbox");
const lightboxImageEl = document.getElementById("lightboxImage");
const lightboxCaptionEl = document.getElementById("lightboxCaption");
const lightboxCloseEl = document.getElementById("lightboxClose");
const batchFilterBarEl = document.getElementById("batchFilterBar");
const themeToggleEl = document.getElementById("themeToggle");
const viewSections = Array.from(document.querySelectorAll("[data-view-section]"));
const navItems = Array.from(document.querySelectorAll("[data-nav-target]"));
let videoStatusTimer = null;
let dashboardTimer = null;
let mobileCameraStream = null;
let mobileCameraTimer = null;
let mobileCameraBusy = false;
let mobileCameraFrameCount = 0;
let mobileCameraFacingMode = "environment";
let mobileCameraStartedAt = null;
let latestBatchResults = [];
let latestVideoLibrary = [];
let selectedVideoLibraryPath = "";
let eventFeedExpanded = false;
let recentTasksExpanded = false;
let latestDemoReport = null;
let latestVisualStrip = null;
const themeStorageKey = "rv1126b-theme";
const defaultView = "overview";
const viewLabels = {
  overview: "工作台",
  mobile: "手机摄像头",
  enhance: "图像增强",
  image: "图片检测",
  video: "视频流检测",
  records: "运行记录",
};

function applyTheme(theme) {
  const normalizedTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = normalizedTheme;
  if (themeToggleEl) {
    themeToggleEl.setAttribute("aria-pressed", String(normalizedTheme === "light"));
    const labelEl = themeToggleEl.querySelector("strong");
    if (labelEl) {
      labelEl.textContent = normalizedTheme === "light" ? "浅色模式" : "深色模式";
    }
  }
}

function initThemeToggle() {
  applyTheme(localStorage.getItem(themeStorageKey) || document.documentElement.dataset.theme || "dark");
  if (!themeToggleEl) {
    return;
  }
  themeToggleEl.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem(themeStorageKey, nextTheme);
    applyTheme(nextTheme);
  });
}

function normalizeView(value) {
  return Object.prototype.hasOwnProperty.call(viewLabels, value) ? value : defaultView;
}

function switchView(nextView, { updateHash = true } = {}) {
  const activeView = normalizeView(nextView);
  viewSections.forEach((section) => {
    const isActive = section.dataset.viewSection === activeView;
    section.hidden = !isActive;
    section.classList.toggle("active", isActive);
  });
  navItems.forEach((item) => {
    const isActive = item.dataset.navTarget === activeView;
    item.classList.toggle("active", isActive);
    item.setAttribute("aria-current", isActive ? "page" : "false");
  });
  if (updateHash) {
    history.replaceState(null, "", `#${activeView}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function initViewNavigation() {
  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      switchView(item.dataset.navTarget || defaultView);
    });
  });
  window.addEventListener("hashchange", () => {
    switchView(window.location.hash.replace("#", ""), { updateHash: false });
  });
  switchView(window.location.hash.replace("#", ""), { updateHash: false });
}

function getCommonFormState() {
  return {
    modelType: document.getElementById("modelType").value,
    threshold: document.getElementById("threshold").value || "0",
    preprocessMode: document.getElementById("preprocessMode").value || "standard",
    saveResults: document.getElementById("saveResults").checked ? "true" : "false",
  };
}

function formatPercent(value) {
  return `${(value * 100).toFixed(4)}%`;
}

function formatDuration(seconds) {
  if (seconds == null) {
    return "--";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(0)} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  const remain = Math.floor(seconds % 60);
  return `${minutes} 分 ${remain} 秒`;
}

function formatSigned(value, digits = 1) {
  if (value == null) {
    return "--";
  }
  const number = Number(value);
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(digits)}`;
}

function verdictBadge(verdict) {
  const cls = verdict === "OK" ? "ok" : "ng";
  return `<span class="badge ${cls}">${verdict}</span>`;
}

function qualityBadge(level) {
  const cls = level === "优秀" ? "ok" : level === "可用" ? "warn" : "ng";
  return `<span class="badge ${cls}">${level}</span>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildPreviewItem(file, index, inputId) {
  const url = URL.createObjectURL(file);
  return `
    <figure class="preview-item" data-preview-url="${url}">
      <button class="preview-remove" type="button" data-preview-remove="${escapeHtml(inputId)}" data-preview-index="${index}" aria-label="删除图片">删除</button>
      <img
        src="${url}"
        alt="已选图片"
        data-lightbox-src="${url}"
        data-lightbox-caption="已选图片预览"
      >
    </figure>
  `;
}

function revokePreviewUrls(container) {
  container.querySelectorAll("[data-preview-url]").forEach((node) => {
    URL.revokeObjectURL(node.dataset.previewUrl);
  });
}

function renderFilePreview(inputEl, targetEl, emptyText) {
  revokePreviewUrls(targetEl);
  const files = inputEl.files;
  if (!files.length) {
    targetEl.classList.add("empty");
    targetEl.textContent = emptyText;
    return;
  }
  targetEl.classList.remove("empty");
  targetEl.innerHTML = Array.from(files).map((file, index) => buildPreviewItem(file, index, inputEl.id)).join("");
}

function removeFileFromInput(inputEl, removeIndex) {
  const transfer = new DataTransfer();
  Array.from(inputEl.files).forEach((file, index) => {
    if (index !== removeIndex) {
      transfer.items.add(file);
    }
  });
  inputEl.files = transfer.files;
}

function renderResultCard(item) {
  return `
    <article class="result-card" data-verdict="${item.verdict}">
      <div class="section-head">
        <div>
          <p class="eyebrow">${item.model_type.toUpperCase()} / ${escapeHtml(item.preprocess_label)}</p>
          <h3>${item.filename}</h3>
        </div>
        <div class="badge-row">
          ${qualityBadge(item.quality_level)}
          ${verdictBadge(item.verdict)}
        </div>
      </div>
      <div class="result-meta">
        <div class="metric">前景像素<strong>${item.foreground_pixels}</strong></div>
        <div class="metric">前景占比<strong>${formatPercent(item.foreground_ratio)}</strong></div>
        <div class="metric">纯推理时间<strong>${item.inference_ms.toFixed(2)} ms</strong></div>
        <div class="metric">推理内存占用<strong>${item.inference_memory_mb.toFixed(2)} MB</strong></div>
      </div>
      <div class="result-meta">
        <div class="metric">画质得分<strong>${item.quality_score.toFixed(1)}</strong></div>
        <div class="metric">亮度 / 对比度<strong>${item.input_brightness.toFixed(1)} / ${item.input_contrast.toFixed(1)}</strong></div>
        <div class="metric">清晰度<strong>${item.input_sharpness.toFixed(1)}</strong></div>
        <div class="metric">饱和度 / 阈值<strong>${Number(item.input_saturation || 0).toFixed(1)} / ${item.threshold}</strong></div>
      </div>
      <div class="result-meta">
        <div class="metric">总耗时<strong>${item.total_ms.toFixed(2)} ms</strong></div>
        <div class="metric">模型常驻内存<strong>${item.resident_memory_mb.toFixed(2)} MB</strong></div>
        <div class="metric">保存状态<strong>${item.saved ? "已保存" : "临时结果"}</strong></div>
        <div class="metric">输入格式<strong>${item.runtime_input_color} / ${item.runtime_input_dtype}</strong></div>
      </div>
      <div class="result-grid">
        <figure>
          <figcaption>原图</figcaption>
          <img src="${item.original_url}" alt="原图" data-lightbox-src="${item.original_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 原图">
        </figure>
        <figure>
          <figcaption>预处理图</figcaption>
          <img src="${item.preprocessed_url}" alt="预处理图" data-lightbox-src="${item.preprocessed_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 预处理图">
        </figure>
        <figure>
          <figcaption>叠加结果</figcaption>
          <img src="${item.overlay_url}" alt="叠加结果" data-lightbox-src="${item.overlay_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 叠加结果">
        </figure>
        <figure>
          <figcaption>彩色掩码</figcaption>
          <img src="${item.mask_color_url}" alt="彩色掩码" data-lightbox-src="${item.mask_color_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 彩色掩码">
        </figure>
      </div>
    </article>
  `;
}

function renderPreprocessCard(item) {
  return `
    <article class="result-card">
      <div class="section-head">
        <div>
          <p class="eyebrow">${escapeHtml(item.preprocess_label)}</p>
          <h3>${item.filename}</h3>
        </div>
        <div class="badge-row">
          ${qualityBadge(item.quality_level)}
        </div>
      </div>
      <div class="result-meta">
        <div class="metric">画质得分<strong>${item.quality_score.toFixed(1)}</strong></div>
        <div class="metric">亮度<strong>${item.input_brightness.toFixed(1)}</strong></div>
        <div class="metric">对比度<strong>${item.input_contrast.toFixed(1)}</strong></div>
        <div class="metric">清晰度<strong>${item.input_sharpness.toFixed(1)}</strong></div>
        <div class="metric">饱和度<strong>${Number(item.input_saturation || 0).toFixed(1)}</strong></div>
      </div>
      <div class="result-meta">
        <div class="metric">综合提升<strong>${formatSigned(item.quality_score_delta)}</strong></div>
        <div class="metric">亮度变化<strong>${formatSigned(item.brightness_delta)}</strong></div>
        <div class="metric">对比度变化<strong>${formatSigned(item.contrast_delta)}</strong></div>
        <div class="metric">清晰度变化<strong>${formatSigned(item.sharpness_delta)}</strong></div>
      </div>
      <div class="result-grid two-up">
        <figure>
          <figcaption>原图</figcaption>
          <img src="${item.original_url}" alt="原图" data-lightbox-src="${item.original_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 原图">
        </figure>
        <figure>
          <figcaption>预处理图</figcaption>
          <img src="${item.preprocessed_url}" alt="预处理图" data-lightbox-src="${item.preprocessed_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 预处理图">
        </figure>
      </div>
    </article>
  `;
}

function renderComparePreprocess(payload) {
  const ranked = [...(payload.results || [])].sort((a, b) => b.quality_score - a.quality_score);
  const cards = ranked.map((item, index) => `
    <article class="compare-card ${index === 0 ? "best" : ""}">
      <div class="compare-card-head">
        <div>
          <span class="rank-chip">${index === 0 ? "推荐" : `#${index + 1}`}</span>
          <h4>${escapeHtml(item.preprocess_label)}</h4>
        </div>
        ${qualityBadge(item.quality_level)}
      </div>
      <div class="compare-image-pair">
        <figure>
          <figcaption>原图</figcaption>
          <img src="${item.original_url}" alt="原图" data-lightbox-src="${item.original_url}" data-lightbox-caption="${escapeHtml(item.preprocess_label)} / 原图">
        </figure>
        <figure>
          <figcaption>增强图</figcaption>
          <img src="${item.preprocessed_url}" alt="增强图" data-lightbox-src="${item.preprocessed_url}" data-lightbox-caption="${escapeHtml(item.preprocess_label)} / 增强图">
        </figure>
      </div>
      <div class="compare-metrics">
        <span>得分 <strong>${item.quality_score.toFixed(1)}</strong></span>
        <span>提升 <strong>${formatSigned(item.quality_score_delta)}</strong></span>
        <span>对比度 <strong>${formatSigned(item.contrast_delta)}</strong></span>
        <span>清晰度 <strong>${formatSigned(item.sharpness_delta)}</strong></span>
      </div>
    </article>
  `).join("");
  return `
    <div class="compare-summary">
      <strong>推荐策略：${escapeHtml(payload.best_label || "--")}</strong>
      <span>根据综合画质得分排序，用于选择当前输入下更稳定的增强策略。</span>
    </div>
    <div class="compare-grid">${cards}</div>
  `;
}

function renderVideoInfoPanel(data) {
  const verdictText = data.verdict || "--";
  const inferenceText = data.inference_ms == null ? "--" : `${Number(data.inference_ms).toFixed(2)} ms`;
  const memoryText = data.inference_memory_mb == null ? "--" : `${Number(data.inference_memory_mb).toFixed(2)} MB`;
  const foregroundText = data.foreground_pixels == null ? "--" : `${data.foreground_pixels}`;
  const fpsText = data.fps_estimate == null ? "--" : `${Number(data.fps_estimate).toFixed(2)}`;
  const qualityText = data.quality_score == null ? "--" : `${Number(data.quality_score).toFixed(1)} / ${escapeHtml(data.quality_level || "--")}`;
  const modelText = data.model_type ? data.model_type.toUpperCase() : "--";
  const preprocessText = data.preprocess_label || "--";
  const stateText = data.message || "未启动";
  videoInfoPanelEl.innerHTML = `
    <div class="metric">状态<strong>${escapeHtml(stateText)}</strong></div>
    <div class="metric">模型<strong>${escapeHtml(modelText)}</strong></div>
    <div class="metric">预处理<strong>${escapeHtml(preprocessText)}</strong></div>
    <div class="metric">判定<strong>${escapeHtml(verdictText)}</strong></div>
    <div class="metric">前景像素<strong>${escapeHtml(foregroundText)}</strong></div>
    <div class="metric">推理时间<strong>${escapeHtml(inferenceText)}</strong></div>
    <div class="metric">估算 FPS<strong>${escapeHtml(fpsText)}</strong></div>
    <div class="metric">画质得分<strong>${qualityText}</strong></div>
  `;
}

function updateDemoPreviewFromResult(item) {
  demoPreviewPanelEl.className = "demo-preview-result";
  demoPreviewPanelEl.innerHTML = `
    <img src="${item.overlay_url}" alt="最近检测叠加图" data-lightbox-src="${item.overlay_url}" data-lightbox-caption="${escapeHtml(item.filename)} / 最近检测叠加图">
    <div class="demo-preview-overlay">
      <strong>${escapeHtml(item.filename)}</strong>
      <span>${escapeHtml(item.model_type.toUpperCase())} / ${escapeHtml(item.preprocess_label)} / ${item.inference_ms.toFixed(2)} ms / ${item.verdict}</span>
    </div>
  `;
  latestVisualStrip = {
    source: "Web 图片上传",
    preprocess: item.preprocess_label || "--",
    model: item.model_type ? item.model_type.toUpperCase() : "--",
    inference: item.inference_ms == null ? "--" : `${Number(item.inference_ms).toFixed(2)} ms`,
    fps: "--",
  };
  renderDemoLiveStrip(latestVisualStrip);
}

function renderDemoLiveStrip({ model = "--", preprocess = "--", source = "--", inference = "--", fps = "--" } = {}) {
  demoLiveStripEl.innerHTML = `
    <span><b>输入源：</b>${escapeHtml(source)}</span>
    <span><b>预处理：</b>${escapeHtml(preprocess)}</span>
    <span><b>模型：</b>${escapeHtml(model)}</span>
    <span><b>推理：</b>${escapeHtml(inference)}</span>
    <span><b>FPS：</b>${escapeHtml(fps)}</span>
  `;
}

function renderMobileCameraStats({
  status = "未启动",
  source = "手机摄像头",
  inference = "--",
  fps = "--",
  verdict = "--",
  frameCount = mobileCameraFrameCount,
} = {}) {
  mobileCameraStatsEl.innerHTML = `
    <div class="metric">状态<strong>${escapeHtml(status)}</strong></div>
    <div class="metric">输入源<strong>${escapeHtml(source)}</strong></div>
    <div class="metric">最新推理<strong>${escapeHtml(inference)}</strong></div>
    <div class="metric">估算 FPS<strong>${escapeHtml(fps)}</strong></div>
    <div class="metric">判定<strong>${escapeHtml(verdict)}</strong></div>
    <div class="metric">累计帧数<strong>${frameCount}</strong></div>
  `;
}

function setMobileCameraStatus(message, stats = {}) {
  mobileCameraStatusEl.textContent = message;
  renderMobileCameraStats(stats);
}

function isMobileCameraSecureContext() {
  return window.isSecureContext || ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

function isMobileCameraSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function setMobileCameraPreviewVisible(active) {
  mobileCameraPlaceholderEl.classList.toggle("hidden", active);
  mobileCameraVideoEl.classList.toggle("active", active);
}

function setMobileCameraResultVisible(active) {
  mobileCameraResultPlaceholderEl.classList.toggle("hidden", active);
  mobileCameraResultEl.classList.toggle("active", active);
}

function canvasToBlob(canvas, type = "image/jpeg", quality = 0.76) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("摄像头帧编码失败"));
      }
    }, type, quality);
  });
}

function updateDemoPreviewFromMobileFrame(payload) {
  demoPreviewPanelEl.className = "demo-preview-result";
  demoPreviewPanelEl.innerHTML = `
    <img src="${payload.frame_data_url}" alt="手机摄像头检测叠加图" data-lightbox-src="${payload.frame_data_url}" data-lightbox-caption="手机摄像头 / 最近检测叠加图">
    <div class="demo-preview-overlay">
      <strong>手机摄像头</strong>
      <span>${escapeHtml(payload.model_type.toUpperCase())} / ${escapeHtml(payload.preprocess_label)} / ${Number(payload.inference_ms).toFixed(2)} ms / ${escapeHtml(payload.verdict)}</span>
    </div>
  `;
  latestVisualStrip = {
    source: "手机摄像头",
    preprocess: payload.preprocess_label || "--",
    model: payload.model_type ? payload.model_type.toUpperCase() : "--",
    inference: payload.inference_ms == null ? "--" : `${Number(payload.inference_ms).toFixed(2)} ms`,
    fps: payload.fps_estimate == null ? "--" : Number(payload.fps_estimate).toFixed(2),
  };
  renderDemoLiveStrip(latestVisualStrip);
}

async function startMobileCamera() {
  if (!isMobileCameraSecureContext()) {
    throw new Error("iPhone Safari 调用摄像头需要 HTTPS 安全来源。安装并信任本地 CA 证书后，请访问 https://100.81.26.139:8443。");
  }
  if (!isMobileCameraSupported()) {
    throw new Error("当前浏览器没有开放摄像头接口，请确认 Safari 权限设置或更换浏览器。");
  }
  stopMobileCameraStream();
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: mobileCameraFacingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };
  mobileCameraStream = await navigator.mediaDevices.getUserMedia(constraints);
  mobileCameraVideoEl.srcObject = mobileCameraStream;
  await mobileCameraVideoEl.play();
  mobileCameraFrameCount = 0;
  mobileCameraStartedAt = Date.now();
  setMobileCameraPreviewVisible(true);
  setMobileCameraResultVisible(false);
  setMobileCameraStatus("摄像头已开启，可开始实时检测。", {
    status: "已开启",
    inference: "--",
    fps: "--",
    verdict: "--",
    frameCount: 0,
  });
}

function stopMobileCameraStream() {
  if (mobileCameraStream) {
    mobileCameraStream.getTracks().forEach((track) => track.stop());
    mobileCameraStream = null;
  }
  mobileCameraVideoEl.pause();
  mobileCameraVideoEl.removeAttribute("srcObject");
  mobileCameraVideoEl.srcObject = null;
  setMobileCameraPreviewVisible(false);
}

async function postMobileCameraFrame() {
  if (!mobileCameraStream || mobileCameraBusy || !mobileCameraVideoEl.videoWidth || !mobileCameraVideoEl.videoHeight) {
    return;
  }
  mobileCameraBusy = true;
  const canvas = mobileCameraCanvasEl;
  const sourceWidth = mobileCameraVideoEl.videoWidth;
  const sourceHeight = mobileCameraVideoEl.videoHeight;
  const maxWidth = 720;
  const scale = Math.min(1, maxWidth / sourceWidth);
  canvas.width = Math.round(sourceWidth * scale);
  canvas.height = Math.round(sourceHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(mobileCameraVideoEl, 0, 0, canvas.width, canvas.height);

  try {
    const blob = await canvasToBlob(canvas);
    const state = getCommonFormState();
    const formData = new FormData();
    formData.append("image", blob, `mobile-camera-${Date.now()}.jpg`);
    formData.append("model_type", state.modelType);
    formData.append("threshold", state.threshold);
    formData.append("preprocess_mode", state.preprocessMode);
    const resp = await fetch("/api/mobile/frame", {
      method: "POST",
      body: formData,
    });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "手机摄像头帧检测失败");
    }
    mobileCameraFrameCount += 1;
    mobileCameraResultEl.src = payload.frame_data_url;
    mobileCameraResultEl.setAttribute("data-lightbox-src", payload.frame_data_url);
    mobileCameraResultEl.setAttribute("data-lightbox-caption", "手机摄像头 / 检测叠加图");
    setMobileCameraResultVisible(true);
    const inferenceText = `${Number(payload.inference_ms).toFixed(2)} ms`;
    const fpsText = Number(payload.fps_estimate).toFixed(2);
    renderMobileCameraStats({
      status: "检测中",
      inference: inferenceText,
      fps: fpsText,
      verdict: payload.verdict || "--",
      frameCount: mobileCameraFrameCount,
    });
    mobileCameraStatusEl.textContent = `检测中：${payload.preprocess_label} / ${inferenceText} / FPS ${fpsText}`;
    updateDemoPreviewFromMobileFrame(payload);
    renderVideoInfoPanel({
      message: "手机摄像头检测进行中",
      model_type: payload.model_type,
      preprocess_label: payload.preprocess_label,
      verdict: payload.verdict,
      foreground_pixels: payload.foreground_pixels,
      inference_ms: payload.inference_ms,
      fps_estimate: payload.fps_estimate,
      quality_score: payload.quality_score,
      quality_level: payload.quality_level,
    });
    if (mobileCameraFrameCount === 1 || mobileCameraFrameCount % 5 === 0) {
      fetchServiceHealth();
      fetchSystemStats();
      fetchRecentEvents();
      fetchDemoOverview();
    }
  } catch (error) {
    mobileCameraStatusEl.textContent = error.message;
  } finally {
    mobileCameraBusy = false;
  }
}

async function startMobileCameraDetection() {
  if (!mobileCameraStream) {
    await startMobileCamera();
  }
  if (mobileCameraTimer) {
    return;
  }
  setMobileCameraStatus("实时检测已启动。", {
    status: "检测中",
    inference: "--",
    fps: "--",
    verdict: "--",
    frameCount: mobileCameraFrameCount,
  });
  await postMobileCameraFrame();
  mobileCameraTimer = window.setInterval(postMobileCameraFrame, 1000);
}

async function stopMobileCameraDetection() {
  if (mobileCameraTimer) {
    window.clearInterval(mobileCameraTimer);
    mobileCameraTimer = null;
  }
  mobileCameraBusy = false;
  stopMobileCameraStream();
  setMobileCameraStatus("手机摄像头检测已停止。", {
    status: "已停止",
    inference: "--",
    fps: "--",
    verdict: "--",
    frameCount: mobileCameraFrameCount,
  });
  try {
    await fetch("/api/mobile/stop", { method: "POST" });
  } catch (error) {
    // 前端已经停止取流，后端状态刷新失败时保持页面可用。
  }
  fetchVideoStatus();
  fetchServiceHealth();
  fetchSystemStats();
}

function renderPipeline(payload) {
  const nodes = payload.nodes || [];
  pipelinePanelEl.innerHTML = nodes.map((node) => `
    <div class="pipeline-node ${escapeHtml(node.status)}">
      <span class="pipeline-dot"></span>
      <div>
        <strong>${escapeHtml(node.title)}</strong>
        <p>${escapeHtml(node.detail)}</p>
      </div>
    </div>
  `).join("");
}

function renderInputMatrix(payload) {
  const items = Object.values(payload || {});
  inputMatrixPanelEl.innerHTML = items.map((item) => `
    <div class="input-source-card">
      <strong>${escapeHtml(item.label)}</strong>
      <span>${escapeHtml(item.status || "--")}</span>
      <p>${escapeHtml(item.latest || item.source || item.description || item.message || "")}</p>
    </div>
  `).join("");
}

function renderPerformanceSummary(data) {
  const speedup = data.int8_speedup > 0 ? `${Number(data.int8_speedup).toFixed(2)}x` : "--";
  const fps = data.current_video_fps == null ? "--" : Number(data.current_video_fps).toFixed(2);
  performancePanelEl.innerHTML = `
    <div class="metric">FP 平均推理<strong>${Number(data.fp_avg_inference_ms || 0).toFixed(2)} ms</strong></div>
    <div class="metric">INT8 平均推理<strong>${Number(data.int8_avg_inference_ms || 0).toFixed(2)} ms</strong></div>
    <div class="metric">INT8 加速比<strong>${speedup}</strong></div>
    <div class="metric">当前 FPS<strong>${escapeHtml(fps)}</strong></div>
  `;
}

function renderStability(data) {
  stabilityPanelEl.innerHTML = `
    <div class="metric">记录状态<strong>${data.active ? "记录中" : "待启动"}</strong></div>
    <div class="metric">记录时长<strong>${formatDuration(Number(data.duration_seconds || 0))}</strong></div>
    <div class="metric">累计帧数<strong>${data.video_frames_total || 0}</strong></div>
    <div class="metric">异常事件<strong>${data.recent_error_count || 0}</strong></div>
  `;
}

function renderRobustnessScenarios(payload) {
  robustnessScenariosEl.innerHTML = (payload.scenarios || []).map((item) => `
    <article class="scenario-card">
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.description)}</p>
      <button class="toggle-btn" type="button" data-scenario-preprocess="${escapeHtml(item.suggested_preprocess)}">使用推荐增强</button>
    </article>
  `).join("");
}

function renderBatchFilters(summary) {
  batchFilterBarEl.innerHTML = `
    <div class="filter-bar">
      <button class="filter-chip active" type="button" data-filter="all">全部 ${summary.total}</button>
      <button class="filter-chip" type="button" data-filter="OK">只看 OK ${summary.ok_count}</button>
      <button class="filter-chip" type="button" data-filter="NG">只看 NG ${summary.ng_count}</button>
    </div>
  `;
  batchFilterBarEl.querySelectorAll(".filter-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      batchFilterBarEl.querySelectorAll(".filter-chip").forEach((node) => node.classList.remove("active"));
      btn.classList.add("active");
      applyBatchFilter(btn.dataset.filter);
    });
  });
}

function applyBatchFilter(filterValue) {
  const cards = document.querySelectorAll("#batchResults .result-card");
  cards.forEach((card) => {
    const verdict = card.getAttribute("data-verdict");
    card.style.display = filterValue === "all" || verdict === filterValue ? "" : "none";
  });
}

function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportBatchCsv(results) {
  const header = [
    "filename",
    "model_type",
    "preprocess_label",
    "verdict",
    "threshold",
    "foreground_pixels",
    "foreground_ratio",
    "quality_score",
    "quality_level",
    "inference_ms",
    "total_ms",
    "resident_memory_mb",
    "inference_memory_mb",
    "saved",
  ];
  const rows = results.map((item) => [
    item.filename,
    item.model_type,
    item.preprocess_label,
    item.verdict,
    item.threshold,
    item.foreground_pixels,
    item.foreground_ratio,
    item.quality_score,
    item.quality_level,
    item.inference_ms,
    item.total_ms,
    item.resident_memory_mb,
    item.inference_memory_mb,
    item.saved,
  ]);
  const csv = [header, ...rows]
    .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  triggerDownload(`seaformer-batch-${Date.now()}.csv`, csv, "text/csv;charset=utf-8");
}

function exportNameList(results, verdict, label) {
  const content = results
    .filter((item) => item.verdict === verdict)
    .map((item) => item.filename)
    .join("\n");
  triggerDownload(`seaformer-${label}-${Date.now()}.txt`, content || "", "text/plain;charset=utf-8");
}

function renderRecentTasks(tasks) {
  if (!tasks.length) {
    recentTasksEl.className = "recent-tasks empty-panel";
    recentTasksEl.textContent = "当前还没有任务记录";
    recentTasksExpanded = false;
    return;
  }
  recentTasksEl.className = "recent-tasks";
  const visible = tasks.slice(0, 1);
  const hidden = tasks.slice(1);
  let html = visible.map((task) => `
    <article class="task-card">
      <div class="task-head">
        <div>
          <strong>${task.task_type === "single" ? "单张任务" : "批量任务"}</strong>
          <div class="task-sub">${new Date(task.created_at).toLocaleString()} / ${task.model_type.toUpperCase()} / ${escapeHtml(task.preprocess_mode)} / 阈值 ${task.threshold}</div>
        </div>
        <span class="badge ${task.ng_count > 0 ? "ng" : "ok"}">${task.ng_count > 0 ? "含 NG" : "全 OK"}</span>
      </div>
      <div class="result-meta">
        <div class="metric">总数<strong>${task.total}</strong></div>
        <div class="metric">OK<strong>${task.ok_count}</strong></div>
        <div class="metric">NG<strong>${task.ng_count}</strong></div>
        <div class="metric">平均推理<strong>${Number(task.avg_inference_ms).toFixed(2)} ms</strong></div>
      </div>
      <div class="task-files"><strong>文件：</strong>${task.filenames.join("、")}</div>
    </article>
  `).join("");
  if (hidden.length) {
    html += `
      <button id="toggleHiddenTasks" class="toggle-btn" type="button">${recentTasksExpanded ? "收起历史任务" : `展开剩余 ${hidden.length} 条任务`}</button>
      <div id="hiddenTasks" class="hidden-results ${recentTasksExpanded ? "open" : ""}">
        ${hidden.map((task) => `
          <article class="task-card">
            <div class="task-head">
              <div>
                <strong>${task.task_type === "single" ? "单张任务" : "批量任务"}</strong>
                <div class="task-sub">${new Date(task.created_at).toLocaleString()} / ${task.model_type.toUpperCase()} / ${escapeHtml(task.preprocess_mode)} / 阈值 ${task.threshold}</div>
              </div>
              <span class="badge ${task.ng_count > 0 ? "ng" : "ok"}">${task.ng_count > 0 ? "含 NG" : "全 OK"}</span>
            </div>
            <div class="result-meta">
              <div class="metric">总数<strong>${task.total}</strong></div>
              <div class="metric">OK<strong>${task.ok_count}</strong></div>
              <div class="metric">NG<strong>${task.ng_count}</strong></div>
              <div class="metric">平均推理<strong>${Number(task.avg_inference_ms).toFixed(2)} ms</strong></div>
            </div>
            <div class="task-files"><strong>文件：</strong>${task.filenames.join("、")}</div>
          </article>
        `).join("")}
      </div>
    `;
  }
  recentTasksEl.innerHTML = html;
  const toggleBtn = document.getElementById("toggleHiddenTasks");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const hiddenBox = document.getElementById("hiddenTasks");
      hiddenBox.classList.toggle("open");
      recentTasksExpanded = hiddenBox.classList.contains("open");
      toggleBtn.textContent = hiddenBox.classList.contains("open")
        ? "收起历史任务"
        : `展开剩余 ${hidden.length} 条任务`;
    });
  } else {
    recentTasksExpanded = false;
  }
}

function renderServiceHealth(data) {
  const loadedModel = data.loaded_model ? data.loaded_model.toUpperCase() : "未加载";
  const apiStatus = data.status === "ok" ? "正常" : "异常";
  serviceHealthPanelEl.innerHTML = `
    <strong>服务状态</strong>
    <div class="service-health-grid">
      <div class="metric">接口状态<strong>${escapeHtml(apiStatus)}</strong></div>
      <div class="metric">当前已加载模型<strong>${escapeHtml(loadedModel)}</strong></div>
      <div class="metric">运行时长<strong>${escapeHtml(formatDuration(Number(data.uptime_seconds || 0)))}</strong></div>
      <div class="metric">累计请求<strong>${escapeHtml(String(data.requests_total || 0))}</strong></div>
    </div>
  `;
}

function renderSystemStats(data) {
  const fpAvg = data.by_model?.fp?.avg_inference_ms ?? 0;
  const int8Avg = data.by_model?.int8?.avg_inference_ms ?? 0;
  systemStatsPanelEl.innerHTML = `
    <div class="metric">累计图片<strong>${data.images_total}</strong></div>
    <div class="metric">平均推理<strong>${Number(data.avg_inference_ms).toFixed(2)} ms</strong></div>
    <div class="metric">NG 数量<strong>${data.ng_count}</strong></div>
    <div class="metric">低质图像<strong>${data.quality_warn_count}</strong></div>
    <div class="metric">视频会话<strong>${data.video_sessions}</strong></div>
    <div class="metric">视频帧数<strong>${data.video_frames}</strong></div>
    <div class="metric">FP 平均推理<strong>${Number(fpAvg).toFixed(2)} ms</strong></div>
    <div class="metric">INT8 平均推理<strong>${Number(int8Avg).toFixed(2)} ms</strong></div>
  `;
}

function renderEventFeed(events) {
  if (!events.length) {
    eventFeedEl.className = "event-feed empty-panel";
    eventFeedEl.textContent = "当前还没有事件记录";
    eventFeedExpanded = false;
    return;
  }
  eventFeedEl.className = "event-feed";
  const visible = events.slice(0, eventPreviewLimit);
  const hidden = events.slice(eventPreviewLimit);
  let html = visible.map((item) => `
    <article class="event-item ${escapeHtml(item.severity)}">
      <div class="event-head">
        <strong>${escapeHtml(item.title)}</strong>
        <span class="badge ${item.severity === "error" ? "ng" : item.severity === "warn" ? "warn" : "ok"}">${escapeHtml(item.severity)}</span>
      </div>
      <div class="event-sub">${new Date(item.created_at).toLocaleString()}</div>
      <p>${escapeHtml(item.detail)}</p>
    </article>
  `).join("");
  if (hidden.length) {
    html += `
      <button id="toggleHiddenEvents" class="toggle-btn" type="button">${eventFeedExpanded ? "收起历史事件" : `展开剩余 ${hidden.length} 条事件`}</button>
      <div id="hiddenEvents" class="hidden-results ${eventFeedExpanded ? "open" : ""}">
        ${hidden.map((item) => `
          <article class="event-item ${escapeHtml(item.severity)}">
            <div class="event-head">
              <strong>${escapeHtml(item.title)}</strong>
              <span class="badge ${item.severity === "error" ? "ng" : item.severity === "warn" ? "warn" : "ok"}">${escapeHtml(item.severity)}</span>
            </div>
            <div class="event-sub">${new Date(item.created_at).toLocaleString()}</div>
            <p>${escapeHtml(item.detail)}</p>
          </article>
        `).join("")}
      </div>
    `;
  }
  eventFeedEl.innerHTML = html;
  const toggleBtn = document.getElementById("toggleHiddenEvents");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const hiddenBox = document.getElementById("hiddenEvents");
      hiddenBox.classList.toggle("open");
      eventFeedExpanded = hiddenBox.classList.contains("open");
      toggleBtn.textContent = hiddenBox.classList.contains("open")
        ? "收起历史事件"
        : `展开剩余 ${hidden.length} 条事件`;
    });
  } else {
    eventFeedExpanded = false;
  }
}

function renderVideoLibrary(videos) {
  latestVideoLibrary = videos;
  const currentValue = selectedVideoLibraryPath;
  if (videos.some((item) => item.path === currentValue)) {
    selectedVideoLibraryPath = currentValue;
  } else if (videos.length) {
    selectedVideoLibraryPath = videos[0].path;
    videoSourceEl.value = videos[0].path;
  } else {
    selectedVideoLibraryPath = "";
    videoSourceEl.value = "";
  }
  videoUploadStatusEl.textContent = videos.length
    ? `当前视频库共有 ${videos.length} 个视频。`
    : "当前视频库没有可用视频。";
  renderVideoLibraryDropdown();
}

function renderVideoLibraryDropdown() {
  if (!latestVideoLibrary.length) {
    videoLibraryTriggerEl.textContent = "当前视频库没有可用视频";
    videoLibraryMenuEl.innerHTML = `<div class="library-empty">暂无视频</div>`;
    return;
  }
  const selectedItem = latestVideoLibrary.find((item) => item.path === selectedVideoLibraryPath) || latestVideoLibrary[0];
  selectedVideoLibraryPath = selectedItem.path;
  videoLibraryTriggerEl.textContent = selectedItem.filename;
  videoLibraryMenuEl.innerHTML = latestVideoLibrary.map((item) => `
    <button
      class="library-option ${item.path === selectedVideoLibraryPath ? "active" : ""}"
      type="button"
      data-video-path="${escapeHtml(item.path)}"
      data-video-name="${escapeHtml(item.filename)}"
    >
      <strong>${escapeHtml(item.filename)}</strong>
      <span>${new Date(item.updated_at).toLocaleString()} / ${(item.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
    </button>
  `).join("");
  videoLibraryMenuEl.querySelectorAll(".library-option").forEach((button) => {
    button.addEventListener("click", () => {
      const path = button.getAttribute("data-video-path") || "";
      const name = button.getAttribute("data-video-name") || "";
      selectedVideoLibraryPath = path;
      videoSourceEl.value = path;
      videoLibraryTriggerEl.textContent = name;
      closeVideoLibraryDropdown();
      renderVideoLibraryDropdown();
      videoUploadStatusEl.textContent = "已选中视频库中的视频。";
    });
  });
}

function openVideoLibraryDropdown() {
  if (!latestVideoLibrary.length) {
    return;
  }
  videoLibraryDropdownEl.classList.add("open");
  videoLibraryMenuEl.classList.remove("hidden");
}

function closeVideoLibraryDropdown() {
  videoLibraryDropdownEl.classList.remove("open");
  videoLibraryMenuEl.classList.add("hidden");
}

function toggleVideoLibraryDropdown() {
  if (videoLibraryDropdownEl.classList.contains("open")) {
    closeVideoLibraryDropdown();
  } else {
    openVideoLibraryDropdown();
  }
}

async function fetchServiceHealth() {
  try {
    const resp = await fetch("/api/health");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "服务状态读取失败");
    }
    renderServiceHealth(payload);
  } catch (error) {
    serviceHealthPanelEl.innerHTML = `
      <strong>服务状态</strong>
      <div class="service-health-grid">
        <div class="metric">接口状态<strong>异常</strong></div>
        <div class="metric">当前已加载模型<strong>--</strong></div>
        <div class="metric">运行时长<strong>--</strong></div>
        <div class="metric">累计请求<strong>--</strong></div>
      </div>
    `;
  }
}

async function fetchSystemStats() {
  try {
    const resp = await fetch("/api/stats");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "系统统计读取失败");
    }
    renderSystemStats(payload);
  } catch (error) {
    systemStatsPanelEl.innerHTML = `<div class="metric">系统统计<strong>读取失败</strong></div>`;
  }
}

async function fetchRecentEvents() {
  try {
    const resp = await fetch("/api/events/recent");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "事件记录读取失败");
    }
    renderEventFeed(payload.events || []);
  } catch (error) {
    eventFeedEl.className = "event-feed empty-panel";
    eventFeedEl.textContent = "事件记录读取失败";
  }
}

async function fetchVideoLibrary() {
  try {
    const resp = await fetch("/api/videos/library");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "视频库读取失败");
    }
    renderVideoLibrary(payload.videos || []);
  } catch (error) {
    videoUploadStatusEl.textContent = error.message;
  }
}

async function fetchRecentTasks() {
  try {
    const resp = await fetch("/api/tasks/recent");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "最近任务读取失败");
    }
    renderRecentTasks(payload.tasks || []);
  } catch (error) {
    recentTasksEl.className = "recent-tasks empty-panel";
    recentTasksEl.textContent = "最近任务读取失败";
  }
}

async function fetchDemoOverview() {
  try {
    const resp = await fetch("/api/demo/overview");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "运行概览读取失败");
    }
    renderPipeline(payload.pipeline || {});
    renderInputMatrix(payload.inputs || {});
    renderPerformanceSummary(payload.performance || {});
    renderStability(payload.stability || {});
    const perf = payload.performance || {};
    const pipeCurrent = payload.pipeline?.current || {};
    if (pipeCurrent.video_active) {
      latestVisualStrip = null;
      renderDemoLiveStrip({
        source: pipeCurrent.video_source || "视频库/摄像头",
        preprocess: pipeCurrent.preprocess_label || "--",
        model: perf.current_model ? perf.current_model.toUpperCase() : "--",
        inference: perf.current_video_inference_ms == null ? "--" : `${Number(perf.current_video_inference_ms).toFixed(2)} ms`,
        fps: perf.current_video_fps == null ? "--" : Number(perf.current_video_fps).toFixed(2),
      });
    } else if (latestVisualStrip) {
      renderDemoLiveStrip(latestVisualStrip);
    } else {
      renderDemoLiveStrip({
        source: pipeCurrent.video_source || "Web/视频库/摄像头",
        preprocess: pipeCurrent.preprocess_label || "--",
        model: perf.current_model ? perf.current_model.toUpperCase() : "--",
        inference: perf.current_video_inference_ms == null ? "--" : `${Number(perf.current_video_inference_ms).toFixed(2)} ms`,
        fps: perf.current_video_fps == null ? "--" : Number(perf.current_video_fps).toFixed(2),
      });
    }
  } catch (error) {
    pipelinePanelEl.innerHTML = `<div class="empty-panel">运行概览读取失败</div>`;
  }
}

async function fetchRobustnessScenarios() {
  try {
    const resp = await fetch("/api/robustness/scenarios");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "复杂场景读取失败");
    }
    renderRobustnessScenarios(payload);
  } catch (error) {
    robustnessScenariosEl.innerHTML = `<div class="empty-panel">复杂场景读取失败</div>`;
  }
}

async function fetchVideoStatus() {
  try {
    const resp = await fetch("/api/video/status");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "视频状态读取失败");
    }
    renderVideoInfoPanel(payload);
  } catch (error) {
    renderVideoInfoPanel({ message: "视频状态读取失败" });
  }
}

function startVideoStatusPolling() {
  stopVideoStatusPolling();
  fetchVideoStatus();
  videoStatusTimer = window.setInterval(fetchVideoStatus, 1000);
}

function stopVideoStatusPolling() {
  if (videoStatusTimer) {
    window.clearInterval(videoStatusTimer);
    videoStatusTimer = null;
  }
}

function startDashboardPolling() {
  stopDashboardPolling();
  dashboardTimer = window.setInterval(() => {
    fetchServiceHealth();
    fetchSystemStats();
    fetchRecentEvents();
    fetchDemoOverview();
  }, 4000);
}

function stopDashboardPolling() {
  if (dashboardTimer) {
    window.clearInterval(dashboardTimer);
    dashboardTimer = null;
  }
}

async function postForm(url, formData, statusEl) {
  statusEl.textContent = "正在推理，请稍候...";
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "请求失败");
  }
  statusEl.textContent = "推理完成。";
  return payload;
}

async function probeVideoSource(source) {
  const resp = await fetch(`/api/video/probe?source=${encodeURIComponent(source)}`);
  const payload = await resp.json();
  if (!resp.ok) {
    throw new Error(payload.error || "视频源检测失败");
  }
  return payload;
}

async function uploadVideoFile(file) {
  const formData = new FormData();
  formData.append("video", file);
  const resp = await fetch("/api/video/upload", {
    method: "POST",
    body: formData,
  });
  const payload = await resp.json();
  if (!resp.ok) {
    throw new Error(payload.error || "视频上传失败");
  }
  return payload;
}

function setVideoPlaceholder(title, description) {
  videoPlaceholderEl.innerHTML = `
    <div class="video-placeholder-card">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(description)}</p>
    </div>
  `;
  videoPlaceholderEl.classList.remove("hidden");
  videoStreamEl.classList.add("idle");
}

function openLightbox(src, caption) {
  lightboxImageEl.src = src;
  lightboxCaptionEl.textContent = caption;
  lightboxEl.classList.remove("hidden");
  lightboxEl.setAttribute("aria-hidden", "false");
}

function closeLightbox() {
  lightboxEl.classList.add("hidden");
  lightboxEl.setAttribute("aria-hidden", "true");
  lightboxImageEl.removeAttribute("src");
  lightboxCaptionEl.textContent = "";
}

preprocessImageInput.addEventListener("change", () => {
  renderFilePreview(preprocessImageInput, preprocessPreviewEl, "尚未选择图片");
});

comparePreprocessImageInput.addEventListener("change", () => {
  renderFilePreview(comparePreprocessImageInput, comparePreprocessPreviewEl, "尚未选择图片");
});

singleImageInput.addEventListener("change", () => {
  renderFilePreview(singleImageInput, singlePreviewEl, "尚未选择图片");
});

batchImagesInput.addEventListener("change", () => {
  renderFilePreview(batchImagesInput, batchPreviewEl, "尚未选择图片");
});

document.getElementById("singleForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const imageInput = document.getElementById("singleImage");
  const statusEl = document.getElementById("singleStatus");
  const resultEl = document.getElementById("singleResult");
  if (!imageInput.files.length) {
    statusEl.textContent = "请先选择图片。";
    return;
  }

  const formData = new FormData();
  const state = getCommonFormState();
  formData.append("image", imageInput.files[0]);
  formData.append("model_type", state.modelType);
  formData.append("threshold", state.threshold);
  formData.append("preprocess_mode", state.preprocessMode);
  formData.append("save_results", state.saveResults);

  try {
    const payload = await postForm("/api/infer", formData, statusEl);
    resultEl.innerHTML = renderResultCard(payload);
    updateDemoPreviewFromResult(payload);
    fetchServiceHealth();
    fetchSystemStats();
    fetchRecentTasks();
    fetchRecentEvents();
    fetchDemoOverview();
  } catch (error) {
    statusEl.textContent = error.message;
    resultEl.innerHTML = "";
  }
});

document.getElementById("preprocessForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const statusEl = document.getElementById("preprocessStatus");
  const resultEl = document.getElementById("preprocessResult");
  if (!preprocessImageInput.files.length) {
    statusEl.textContent = "请先选择图片。";
    return;
  }

  const formData = new FormData();
  const state = getCommonFormState();
  formData.append("image", preprocessImageInput.files[0]);
  formData.append("preprocess_mode", state.preprocessMode);

  try {
    statusEl.textContent = "正在生成预处理结果...";
    const response = await fetch("/api/preprocess/preview", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "预处理预览失败");
    }
    statusEl.textContent = "预处理结果已生成。";
    resultEl.innerHTML = renderPreprocessCard(payload);
  } catch (error) {
    statusEl.textContent = error.message;
    resultEl.innerHTML = "";
  }
});

document.getElementById("comparePreprocessForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const statusEl = document.getElementById("comparePreprocessStatus");
  const resultEl = document.getElementById("comparePreprocessResult");
  if (!comparePreprocessImageInput.files.length) {
    statusEl.textContent = "请先选择图片。";
    return;
  }

  const formData = new FormData();
  formData.append("image", comparePreprocessImageInput.files[0]);

  try {
    statusEl.textContent = "正在运行全部预处理策略...";
    const response = await fetch("/api/preprocess/compare", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "多策略对比失败");
    }
    statusEl.textContent = `对比完成，推荐策略：${payload.best_label || "--"}。`;
    resultEl.innerHTML = renderComparePreprocess(payload);
  } catch (error) {
    statusEl.textContent = error.message;
    resultEl.innerHTML = "";
  }
});

document.getElementById("batchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const imageInput = document.getElementById("batchImages");
  const statusEl = document.getElementById("batchStatus");
  const summaryEl = document.getElementById("batchSummary");
  const resultEl = document.getElementById("batchResults");
  if (!imageInput.files.length) {
    statusEl.textContent = "请先选择至少一张图片。";
    return;
  }

  const formData = new FormData();
  const state = getCommonFormState();
  for (const file of imageInput.files) {
    formData.append("images", file);
  }
  formData.append("model_type", state.modelType);
  formData.append("threshold", state.threshold);
  formData.append("preprocess_mode", state.preprocessMode);
  formData.append("save_results", state.saveResults);

  try {
    const payload = await postForm("/api/infer/batch", formData, statusEl);
    const { summary, results } = payload;
    latestBatchResults = results;
    const visible = results.slice(0, previewLimit);
    const hidden = results.slice(previewLimit);
    renderBatchFilters(summary);

    summaryEl.innerHTML = `
      <div class="summary-box">
        <div class="result-meta">
          <div class="metric">总数<strong>${summary.total}</strong></div>
          <div class="metric">OK<strong>${summary.ok_count}</strong></div>
          <div class="metric">NG<strong>${summary.ng_count}</strong></div>
          <div class="metric">平均推理<strong>${summary.avg_inference_ms.toFixed(2)} ms</strong></div>
        </div>
        <div class="result-meta">
          <div class="metric">预处理<strong>${escapeHtml(state.preprocessMode)}</strong></div>
          <div class="metric">低质图像<strong>${summary.quality_warn_count}</strong></div>
          <div class="metric">默认展开<strong>${previewLimit}</strong></div>
          <div class="metric">阈值<strong>${state.threshold}</strong></div>
        </div>
        <div class="ok-list">
          <strong>OK 文件名：</strong>
          <div>${summary.ok_filenames.length ? summary.ok_filenames.join("、") : "无"}</div>
        </div>
        <div class="copy-btn-row">
          <button id="copyOkNamesBtn" class="toggle-btn" type="button">复制 OK 文件名</button>
          <button id="exportCsvBtn" class="toggle-btn" type="button">导出 CSV</button>
          <button id="exportOkTxtBtn" class="toggle-btn" type="button">导出 OK 文件名</button>
          <button id="exportNgTxtBtn" class="toggle-btn" type="button">导出 NG 文件名</button>
        </div>
      </div>
    `;

    let html = visible.map(renderResultCard).join("");
    if (hidden.length) {
      const hiddenHtml = hidden.map(renderResultCard).join("");
      html += `
        <button id="toggleHiddenResults" class="toggle-btn" type="button">展开剩余 ${hidden.length} 项</button>
        <div id="hiddenResults" class="hidden-results">${hiddenHtml}</div>
      `;
    }
    resultEl.innerHTML = html;

    const toggleBtn = document.getElementById("toggleHiddenResults");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const hiddenBox = document.getElementById("hiddenResults");
        hiddenBox.classList.toggle("open");
        toggleBtn.textContent = hiddenBox.classList.contains("open")
          ? "收起剩余结果"
          : `展开剩余 ${hidden.length} 项`;
      });
    }

    const copyBtn = document.getElementById("copyOkNamesBtn");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        const text = summary.ok_filenames.join("\n");
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = "已复制";
          setTimeout(() => {
            copyBtn.textContent = "复制 OK 文件名";
          }, 1200);
        } catch (error) {
          copyBtn.textContent = "复制失败";
          setTimeout(() => {
            copyBtn.textContent = "复制 OK 文件名";
          }, 1200);
        }
      });
    }

    const exportCsvBtn = document.getElementById("exportCsvBtn");
    if (exportCsvBtn) {
      exportCsvBtn.addEventListener("click", () => exportBatchCsv(latestBatchResults));
    }

    const exportOkTxtBtn = document.getElementById("exportOkTxtBtn");
    if (exportOkTxtBtn) {
      exportOkTxtBtn.addEventListener("click", () => exportNameList(latestBatchResults, "OK", "ok-files"));
    }

    const exportNgTxtBtn = document.getElementById("exportNgTxtBtn");
    if (exportNgTxtBtn) {
      exportNgTxtBtn.addEventListener("click", () => exportNameList(latestBatchResults, "NG", "ng-files"));
    }

    fetchRecentTasks();
    fetchRecentEvents();
    fetchSystemStats();
    fetchServiceHealth();
    fetchDemoOverview();
  } catch (error) {
    statusEl.textContent = error.message;
    summaryEl.innerHTML = "";
    batchFilterBarEl.innerHTML = "";
    resultEl.innerHTML = "";
  }
});

document.getElementById("videoProbeBtn").addEventListener("click", async () => {
  const source = videoSourceEl.value.trim();
  if (!source) {
    videoStatusEl.textContent = "请先输入视频源地址或摄像头编号。";
    return;
  }
  videoStatusEl.textContent = "正在检测视频源...";
  try {
    const payload = await probeVideoSource(source);
    setVideoPlaceholder("视频源可用", `分辨率 ${payload.width}x${payload.height}，FPS ${payload.fps.toFixed(2)}。点击“开始视频检测”即可查看实时叠加结果。`);
    videoStatusEl.textContent = `视频源可用，分辨率 ${payload.width}x${payload.height}，FPS ${payload.fps.toFixed(2)}`;
    fetchVideoStatus();
    fetchServiceHealth();
    fetchRecentEvents();
  } catch (error) {
    setVideoPlaceholder("视频源不可用", error.message);
    videoStatusEl.textContent = error.message;
    fetchVideoStatus();
    fetchRecentEvents();
  }
});

document.getElementById("videoStartBtn").addEventListener("click", async () => {
  const source = videoSourceEl.value.trim();
  if (!source) {
    videoStatusEl.textContent = "请先输入视频源地址或摄像头编号。";
    return;
  }
  videoStatusEl.textContent = "正在准备视频检测...";
  try {
    const payload = await probeVideoSource(source);
    setVideoPlaceholder("视频检测启动中", "正在连接视频流并等待第一帧返回，请稍候。");
    const state = getCommonFormState();
    const params = new URLSearchParams({
      source,
      model_type: state.modelType,
      threshold: state.threshold,
      preprocess_mode: state.preprocessMode,
    });
    videoStreamEl.src = `/api/video/stream?${params.toString()}`;
    videoStatusEl.textContent = `视频检测已启动，当前模型 ${state.modelType.toUpperCase()}，分辨率 ${payload.width}x${payload.height}。`;
    startVideoStatusPolling();
    fetchServiceHealth();
    fetchSystemStats();
    fetchDemoOverview();
  } catch (error) {
    setVideoPlaceholder("视频检测启动失败", error.message);
    videoStatusEl.textContent = error.message;
    fetchVideoStatus();
  }
});

document.getElementById("videoStopBtn").addEventListener("click", async () => {
  videoStreamEl.removeAttribute("src");
  setVideoPlaceholder("视频检测已停止", "你可以重新输入视频源并再次启动检测。");
  videoStatusEl.textContent = "视频检测已停止。";
  stopVideoStatusPolling();
  try {
    await fetch("/api/video/stop", { method: "POST" });
  } catch (error) {
    // 手动停止的核心体验是切断前端流，后端状态刷新失败时保持页面可用。
  }
  fetchVideoStatus();
  fetchServiceHealth();
});

document.getElementById("videoLibraryRefreshBtn").addEventListener("click", () => {
  fetchVideoLibrary();
});

document.getElementById("videoUseSelectedBtn").addEventListener("click", () => {
  if (!selectedVideoLibraryPath) {
    videoUploadStatusEl.textContent = "请先从视频库中选择一个视频。";
    return;
  }
  videoSourceEl.value = selectedVideoLibraryPath;
  videoUploadStatusEl.textContent = "已将选中视频填入视频源。";
});

videoLibraryTriggerEl.addEventListener("click", () => {
  toggleVideoLibraryDropdown();
});

document.getElementById("videoUploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!videoUploadInputEl.files.length) {
    videoUploadStatusEl.textContent = "请先选择一个视频文件。";
    return;
  }
  videoUploadStatusEl.textContent = "正在上传并校验视频...";
  try {
    const payload = await uploadVideoFile(videoUploadInputEl.files[0]);
    videoUploadStatusEl.textContent = `视频上传成功：${payload.filename}，分辨率 ${payload.width}x${payload.height}，FPS ${Number(payload.fps).toFixed(2)}`;
    await fetchVideoLibrary();
    selectedVideoLibraryPath = payload.path;
    videoSourceEl.value = payload.path;
    renderVideoLibraryDropdown();
    fetchRecentEvents();
  } catch (error) {
    videoUploadStatusEl.textContent = error.message;
  }
});

document.getElementById("videoSnapshotBtn").addEventListener("click", async () => {
  try {
    const resp = await fetch("/api/video/snapshot");
    if (!resp.ok) {
      const payload = await resp.json();
      throw new Error(payload.error || "当前没有可保存的视频帧");
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `video-snapshot-${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    videoStatusEl.textContent = "当前帧快照已保存到本地下载目录。";
  } catch (error) {
    videoStatusEl.textContent = error.message;
  }
});

document.getElementById("mobileCameraStartBtn").addEventListener("click", async () => {
  try {
    await startMobileCamera();
  } catch (error) {
    setMobileCameraStatus(error.message, {
      status: "不可用",
      inference: "--",
      fps: "--",
      verdict: "--",
      frameCount: mobileCameraFrameCount,
    });
  }
});

document.getElementById("mobileCameraRunBtn").addEventListener("click", async () => {
  try {
    await startMobileCameraDetection();
  } catch (error) {
    setMobileCameraStatus(error.message, {
      status: "不可用",
      inference: "--",
      fps: "--",
      verdict: "--",
      frameCount: mobileCameraFrameCount,
    });
  }
});

document.getElementById("mobileCameraSwitchBtn").addEventListener("click", async () => {
  const wasDetecting = Boolean(mobileCameraTimer);
  if (mobileCameraTimer) {
    window.clearInterval(mobileCameraTimer);
    mobileCameraTimer = null;
  }
  mobileCameraFacingMode = mobileCameraFacingMode === "environment" ? "user" : "environment";
  try {
    await startMobileCamera();
    mobileCameraStatusEl.textContent = mobileCameraFacingMode === "environment"
      ? "已切换到后置摄像头。"
      : "已切换到前置摄像头。";
    if (wasDetecting) {
      await startMobileCameraDetection();
    }
  } catch (error) {
    setMobileCameraStatus(error.message, {
      status: "不可用",
      inference: "--",
      fps: "--",
      verdict: "--",
      frameCount: mobileCameraFrameCount,
    });
  }
});

document.getElementById("mobileCameraStopBtn").addEventListener("click", () => {
  stopMobileCameraDetection();
});

document.getElementById("stabilityStartBtn").addEventListener("click", async () => {
  try {
    const resp = await fetch("/api/stability/start", { method: "POST" });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "稳定性记录启动失败");
    }
    renderStability(payload);
    exportStatusEl.textContent = "稳定性记录已开始。";
    fetchRecentEvents();
  } catch (error) {
    exportStatusEl.textContent = error.message;
  }
});

document.getElementById("stabilityStopBtn").addEventListener("click", async () => {
  try {
    const resp = await fetch("/api/stability/stop", { method: "POST" });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "稳定性记录停止失败");
    }
    renderStability(payload);
    exportStatusEl.textContent = "稳定性记录已停止。";
    fetchRecentEvents();
  } catch (error) {
    exportStatusEl.textContent = error.message;
  }
});

document.getElementById("exportDemoReportBtn").addEventListener("click", async () => {
  exportStatusEl.textContent = "正在生成运行报告...";
  try {
    const resp = await fetch("/api/export/demo-report");
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || "运行报告导出失败");
    }
    latestDemoReport = payload;
    triggerDownload(`rv1126b-runtime-report-${Date.now()}.json`, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
    exportStatusEl.textContent = "运行报告已生成并下载。";
  } catch (error) {
    exportStatusEl.textContent = error.message;
  }
});

robustnessScenariosEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const preprocessMode = target.getAttribute("data-scenario-preprocess");
  if (!preprocessMode) {
    return;
  }
  const select = document.getElementById("preprocessMode");
  select.value = preprocessMode;
  document.getElementById("preprocessStatus").textContent = "已切换为该场景推荐增强策略，可上传样本查看效果。";
});

videoStreamEl.addEventListener("load", () => {
  videoPlaceholderEl.classList.add("hidden");
  videoStreamEl.classList.remove("idle");
});

videoStreamEl.addEventListener("error", () => {
  setVideoPlaceholder("视频流连接中断", "当前视频检测流没有返回可显示画面，请检查视频源或重新启动检测。");
  videoStatusEl.textContent = "视频流连接中断。";
  fetchVideoStatus();
  fetchRecentEvents();
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const previewInputId = target.getAttribute("data-preview-remove");
  if (previewInputId) {
    const previewIndex = Number(target.getAttribute("data-preview-index") || "-1");
    const inputEl = document.getElementById(previewInputId);
    if (inputEl instanceof HTMLInputElement && previewIndex >= 0) {
      removeFileFromInput(inputEl, previewIndex);
      if (inputEl === preprocessImageInput) {
        renderFilePreview(preprocessImageInput, preprocessPreviewEl, "尚未选择图片");
      } else if (inputEl === comparePreprocessImageInput) {
        renderFilePreview(comparePreprocessImageInput, comparePreprocessPreviewEl, "尚未选择图片");
      } else if (inputEl === singleImageInput) {
        renderFilePreview(singleImageInput, singlePreviewEl, "尚未选择图片");
      } else if (inputEl === batchImagesInput) {
        renderFilePreview(batchImagesInput, batchPreviewEl, "尚未选择图片");
      }
    }
    return;
  }
  if (!videoLibraryDropdownEl.contains(target)) {
    closeVideoLibraryDropdown();
  }
  const src = target.getAttribute("data-lightbox-src");
  if (src) {
    openLightbox(src, target.getAttribute("data-lightbox-caption") || "");
  }
});

lightboxCloseEl.addEventListener("click", closeLightbox);

lightboxEl.addEventListener("click", (event) => {
  if (event.target === lightboxEl) {
    closeLightbox();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightboxEl.classList.contains("hidden")) {
    closeLightbox();
  }
});

window.addEventListener("beforeunload", () => {
  if (mobileCameraTimer) {
    window.clearInterval(mobileCameraTimer);
  }
  stopMobileCameraStream();
});

initThemeToggle();
initViewNavigation();
setVideoPlaceholder("视频检测待启动", "输入视频源后点击“开始视频检测”，这里会显示带检测叠加信息的实时画面。");
videoStreamEl.classList.add("idle");
if (!isMobileCameraSecureContext()) {
  setMobileCameraStatus("iPhone Safari 调用摄像头需要 HTTPS 安全来源。安装并信任本地 CA 证书后，请访问 https://100.81.26.139:8443。", {
    status: "不可用",
    inference: "--",
    fps: "--",
    verdict: "--",
    frameCount: 0,
  });
} else if (!isMobileCameraSupported()) {
  setMobileCameraStatus("当前浏览器没有开放摄像头接口，请确认 Safari 权限设置或更换浏览器。", {
    status: "不可用",
    inference: "--",
    fps: "--",
    verdict: "--",
    frameCount: 0,
  });
} else {
  renderMobileCameraStats();
}
fetchVideoStatus();
fetchServiceHealth();
fetchSystemStats();
fetchRecentEvents();
fetchRecentTasks();
fetchVideoLibrary();
fetchDemoOverview();
fetchRobustnessScenarios();
startDashboardPolling();
