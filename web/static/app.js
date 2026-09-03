(() => {
  const $ = (id) => document.getElementById(id);

  let selectedFile = null;
  let activeSample = null;
  let pollTimer = null;

  const fileInput = $("fileInput");
  const runBtn = $("runBtn");
  const fileName = $("fileName");

  fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files?.[0] || null;
    activeSample = null;
    fileName.textContent = selectedFile ? selectedFile.name : "尚未选择文件";
    runBtn.disabled = !selectedFile;
    if (selectedFile && /\.(mp4|mkv|mov|webm)$/i.test(selectedFile.name)) {
      $("forceVideo").checked = true;
      if (!$("profile").value) $("profile").value = "video_talking";
      if (!$("sourceLang").value) $("sourceLang").value = "en";
      $("targetLang").value = "zh";
    }
  });

  runBtn.addEventListener("click", () => startJob());

  async function loadSamples() {
    const box = $("sampleList");
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      box.innerHTML = "";
      for (const s of data.samples || []) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "btn small" + (s.available ? "" : " unavailable");
        b.textContent = s.label;
        b.disabled = !s.available;
        b.addEventListener("click", () => {
          selectedFile = null;
          fileInput.value = "";
          activeSample = s;
          fileName.textContent = `样例：${s.label}`;
          runBtn.disabled = false;
          $("targetLang").value = s.target_lang || "en";
          $("sourceLang").value = s.source_lang || "";
          $("profile").value = s.profile || "";
          $("singing").checked = !!s.singing;
          $("assumeSingle").checked = !!s.assume_single;
          $("forceVideo").checked = !!s.video;
          $("noGold").checked = s.no_gold !== false;
          $("noLyricsHint").checked = !!s.no_lyrics_hint;
        });
        box.appendChild(b);
      }
    } catch (e) {
      box.innerHTML = `<p class="muted">样例加载失败：${e}</p>`;
    }
  }

  async function health() {
    try {
      const r = await fetch("/api/health");
      const d = await r.json();
      $("healthHint").textContent = d.pipeline_loaded
        ? "模型已加载"
        : "模型将在首次任务时加载";
    } catch {
      $("healthHint").textContent = "后端未连接";
    }
  }

  function formParams() {
    const fd = new FormData();
    fd.append("target_lang", $("targetLang").value);
    if ($("sourceLang").value) fd.append("source_lang", $("sourceLang").value);
    if ($("profile").value) fd.append("profile", $("profile").value);
    fd.append("device", $("device").value);
    fd.append("singing", $("singing").checked ? "true" : "false");
    fd.append("assume_single", $("assumeSingle").checked ? "true" : "false");
    fd.append("no_gold", $("noGold").checked ? "true" : "false");
    fd.append("no_lyrics_hint", $("noLyricsHint").checked ? "true" : "false");
    fd.append("video", $("forceVideo").checked ? "true" : "false");
    return fd;
  }

  async function startJob() {
    const fd = formParams();
    if (activeSample) {
      fd.append("sample_id", activeSample.id);
    } else if (selectedFile) {
      fd.append("file", selectedFile);
    } else {
      return;
    }

    $("statusPanel").hidden = false;
    $("resultPanel").hidden = true;
    $("statusPill").textContent = "queued";
    $("statusPill").className = "pill";
    $("statusMsg").textContent = "提交中…";
    $("progressBar").className = "progress-bar indeterminate";
    $("progressBar").style.width = "";
    runBtn.disabled = true;

    try {
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }
      const data = await res.json();
      $("jobId").textContent = data.job_id;
      pollJob(data.job_id);
    } catch (e) {
      $("statusPill").textContent = "error";
      $("statusPill").className = "pill error";
      $("statusMsg").textContent = String(e.message || e);
      $("progressBar").className = "progress-bar";
      runBtn.disabled = false;
    }
  }

  function pollJob(id) {
    if (pollTimer) clearInterval(pollTimer);
    const tick = async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`);
        const job = await res.json();
        $("statusPill").textContent = job.status;
        $("statusPill").className = "pill " + (job.status || "");
        $("statusMsg").textContent = job.message || "";
        if (job.status === "running" || job.status === "queued") {
          $("progressBar").className = "progress-bar indeterminate";
          return;
        }
        clearInterval(pollTimer);
        pollTimer = null;
        $("progressBar").className = "progress-bar";
        $("progressBar").style.width = "100%";
        runBtn.disabled = false;
        if (job.status === "success" || job.status === "degraded") {
          showResult(job);
        }
      } catch (e) {
        $("statusMsg").textContent = "轮询失败: " + e;
      }
    };
    tick();
    pollTimer = setInterval(tick, 2000);
  }

  function showResult(job) {
    const r = job.result || {};
    $("resultPanel").hidden = false;
    $("metrics").innerHTML = [
      metric("状态", r.status || job.status),
      metric("耗时", (r.processing_time ?? "—") + "s"),
      metric("语种", `${r.source_lang || "?"} → ${r.target_lang || "?"}`),
      metric("sync", r.sync_score != null ? Number(r.sync_score).toFixed(2) : "—"),
      metric("profile", r.profile || "—"),
    ].join("");
    $("asrText").textContent = r.asr_text || "（无）";
    $("trText").textContent = stripTranslationHeader(r.translation) || "（无）";
    $("summaryPre").textContent = r.summary || "";

    const audio = $("audioPlayer");
    const wavLink = $("wavLink");
    if (r.has_wav) {
      const url = `/api/jobs/${job.id}/file/wav?t=${Date.now()}`;
      audio.src = url;
      wavLink.href = url;
      wavLink.hidden = false;
    } else {
      audio.removeAttribute("src");
      wavLink.hidden = true;
    }

    const wrap = $("videoWrap");
    if (r.has_video) {
      wrap.hidden = false;
      const url = `/api/jobs/${job.id}/file/video?t=${Date.now()}`;
      $("videoPlayer").src = url;
      $("videoLink").href = url;
    } else {
      wrap.hidden = true;
      $("videoPlayer").removeAttribute("src");
    }

    $("resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function metric(k, v) {
    return `<span><strong>${k}</strong> ${escapeHtml(String(v))}</span>`;
  }

  function stripTranslationHeader(t) {
    if (!t) return "";
    // translation.txt 可能含标题块，尽量抽译文段落
    const m = t.match(/\[译文[^\]]*\]\s*([\s\S]*?)(?:\n情感:|$)/);
    if (m) return m[1].trim();
    return t.trim();
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  loadSamples();
  health();
})();
