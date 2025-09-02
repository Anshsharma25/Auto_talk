// frontend/app.js
const imgInput = document.getElementById("imgInput");
const facePreview = document.getElementById("facePreview");
const recBtn = document.getElementById("recBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const replyVideo = document.getElementById("replyVideo");
const replyAudio = document.getElementById("replyAudio");
const micCheckBtn = document.getElementById("micCheckBtn");
const micSelect = document.getElementById("micSelect");

let audioCtx = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let recordedChunks = [];
let recording = false;
let currentImageFile = null;

function log(msg){
  const d = document.createElement("div");
  d.textContent = msg;
  logEl.prepend(d);
}

imgInput.addEventListener("change", (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  currentImageFile = f;
  facePreview.src = URL.createObjectURL(f);
  log("Image selected: " + f.name);
});

async function populateMicList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter(d => d.kind === "audioinput");
    micSelect.innerHTML = "";
    mics.forEach(m=>{
      const opt = document.createElement("option");
      opt.value = m.deviceId;
      opt.textContent = m.label || `Microphone (${m.deviceId.slice(0,6)})`;
      micSelect.appendChild(opt);
    });
    if(mics.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No microphone found";
      micSelect.appendChild(opt);
    }
  } catch(err){
    console.warn("enumerateDevices error", err);
  }
}

micCheckBtn && micCheckBtn.addEventListener("click", async ()=>{
  try {
    statusEl.textContent = "Requesting mic permission...";
    await navigator.mediaDevices.getUserMedia({ audio: true });
    await populateMicList();
    statusEl.textContent = "Mic permission granted";
  } catch(err){
    console.error(err);
    statusEl.textContent = "Mic permission denied or unavailable";
    alert("Microphone access denied or unavailable. Check your browser permissions.");
  }
});

recBtn.addEventListener("click", async ()=>{
  if(!currentImageFile){ alert("Please upload an image first."); return; }
  if (recording) return;
  try {
    const deviceId = micSelect && micSelect.value ? micSelect.value : undefined;
    const constraints = { audio: deviceId ? { deviceId: { exact: deviceId } } : true };
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);

    const bufferSize = 4096;
    processorNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    recordedChunks = [];
    processorNode.onaudioprocess = (evt) => {
      if (!recording) return;
      const inputBuffer = evt.inputBuffer.getChannelData(0);
      recordedChunks.push(new Float32Array(inputBuffer));
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    recording = true;
    recBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = "Recording...";
    log("Recording started");
  } catch(err) {
    console.error(err);
    alert("Error starting recording: " + (err.message || err));
    statusEl.textContent = "Record error";
  }
});

stopBtn.addEventListener("click", async ()=>{
  if(!recording) return;
  recording = false;
  recBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Encoding audio...";
  log("Recording stopped - encoding WAV...");

  try {
    if (processorNode) {
      processorNode.disconnect();
      processorNode.onaudioprocess = null;
      processorNode = null;
    }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (audioCtx) { await audioCtx.close(); audioCtx = null; }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
  } catch(e){
    console.warn("Error stopping audio graph", e);
  }

  try {
    const detectedSampleRate = (window._detectedSampleRate) ? window._detectedSampleRate : 48000;
    const merged = mergeFloat32Arrays(recordedChunks);
    const outSampleRate = 16000;
    const down = downsampleBuffer(merged, detectedSampleRate, outSampleRate);
    const wavBuffer = encodeWAV(down, outSampleRate);
    const blob = new Blob([wavBuffer], { type: "audio/wav" });

    statusEl.textContent = "Uploading...";
    log("Uploading WAV (" + Math.round(blob.size/1024) + " KB) to server...");
    await sendToServer(blob);
  } catch(err) {
    console.error(err);
    statusEl.textContent = "Encode/upload error";
    alert("Failed to encode or upload audio: " + (err.message || err));
  }
});

function mergeFloat32Arrays(chunks) {
  if (chunks.length === 0) return new Float32Array(0);
  let length = 0;
  for (let i = 0; i < chunks.length; i++) length += chunks[i].length;
  const result = new Float32Array(length);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    result.set(chunks[i], offset);
    offset += chunks[i].length;
  }
  if (window._detectedSampleRate === undefined) {
    window._detectedSampleRate = 48000;
  }
  return result;
}

function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate === sampleRate) {
    return buffer;
  }
  if (outSampleRate > sampleRate) {
    const result = new Float32Array(Math.round(buffer.length * outSampleRate / sampleRate));
    for (let i = 0; i < result.length; i++) {
      const idx = i * (sampleRate / outSampleRate);
      result[i] = buffer[Math.floor(idx)] || 0;
    }
    return result;
  }
  const ratio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.floor((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = (count > 0) ? (accum / count) : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);
  return view;
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7FFF;
    output.setInt16(offset, s, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

async function sendToServer(wavBlob) {
  try {
    const fd = new FormData();
    fd.append("image", currentImageFile);
    fd.append("audio", wavBlob, "question.wav");

    statusEl.textContent = "Processing on server...";
    log("Sending to server...");

    const resp = await fetch("/talk", { method: "POST", body: fd });
    if(!resp.ok){
      const j = await resp.json().catch(()=>({error: "unknown"}));
      log("Server error: " + JSON.stringify(j));
      statusEl.textContent = "Error";
      alert("Server returned an error: " + JSON.stringify(j));
      return;
    }
    const data = await resp.json();
    log("Reply text: " + data.reply_text);
    statusEl.textContent = "Received reply";

    const audioBlob = b64ToBlob(data.audio_b64, "audio/wav");
    replyAudio.src = URL.createObjectURL(audioBlob);
    replyAudio.play().catch(()=>{});

    const videoBlob = b64ToBlob(data.video_b64, "video/mp4");
    replyVideo.src = URL.createObjectURL(videoBlob);
    replyVideo.play().catch(()=>{});

  } catch(err){
    console.error(err); log("Network error: " + err.message);
    statusEl.textContent = "Network error";
    alert("Network error: " + (err.message || err));
  }
}

function b64ToBlob(b64, mime){
  const byteChars = atob(b64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], {type: mime});
}

window.addEventListener("load", async ()=>{
  try { await populateMicList(); } catch(e){ /* ignore */ }
});
