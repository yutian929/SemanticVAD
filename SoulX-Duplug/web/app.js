/**
 * SoulX-Duplug browser client (pure client-side; the server is untouched).
 *
 * Pipeline:
 *   getUserMedia -> AudioWorklet(pcm-capture) -> [resample to 16 kHz]
 *   -> 2560-sample chunks (160 ms) -> base64(float32 LE) -> WebSocket /turn
 *   -> server replies {type:"turn_state", state:{state, text, asr_segment, asr_buffer}}
 *
 * This mirrors exactly what test.py / example_client.py send, so no server-side
 * change is required.
 */

const TARGET_SR = 16000;
const CHUNK = 2560; // 160 ms @ 16 kHz, must match config/config.yaml input.chunk_size
const DEFAULT_WS = 'ws://localhost:8000/turn';
const LS_KEY = 'soulx-duplug-ws-url';

const STATE_META = {
  idle: {
    zh: '静默',
    en: 'idle',
    desc: '当前分块无语义内容（静音 / 噪声 / 附和语），系统继续聆听。',
  },
  nonidle: {
    zh: '说话中',
    en: 'nonidle',
    desc: '当前分块含语义内容，用户正在说话，尚未判定说完。',
  },
  speak: {
    zh: '可接话',
    en: 'speak',
    desc: '判定用户已停止说话且语义完整 —— 系统可以接管发言权。',
  },
  blank: {
    zh: '缓存中',
    en: 'blank',
    desc: '未处理音频不足一个分块（160ms），服务端已缓存，等待下一次输入。',
  },
};

/* --------------------------- DOM --------------------------- */
const $ = (id) => document.getElementById(id);
const el = {
  serverUrl: $('serverUrl'),
  sessionId: $('sessionId'),
  connPill: $('connPill'),
  connText: $('connText'),
  stateCard: $('stateCard'),
  stateName: $('stateName'),
  stateEn: $('stateEn'),
  stateDesc: $('stateDesc'),
  wave: $('waveCanvas'),
  timeline: $('timelineCanvas'),
  meterFill: $('meterFill'),
  rmsText: $('rmsText'),
  rateTag: $('rateTag'),
  btnStart: $('btnStart'),
  btnStop: $('btnStop'),
  btnReset: $('btnReset'),
  micSelect: $('micSelect'),
  asrSegment: $('asrSegment'),
  asrBuffer: $('asrBuffer'),
  turnList: $('turnList'),
  turnTag: $('turnTag'),
  stSent: $('stSent'),
  stRecv: $('stRecv'),
  stRtt: $('stRtt'),
  stDur: $('stDur'),
  stIdle: $('stIdle'),
  stRate: $('stRate'),
  log: $('log'),
  btnClearLog: $('btnClearLog'),
};

/* --------------------------- runtime state --------------------------- */
let ws = null;
let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let scriptNode = null; // fallback when AudioWorklet is unavailable
let analyser = null;
let resampler = null;
let pending = new Float32Array(0); // 16 kHz samples not yet sent
let running = false;
let startTs = 0;

const sendTimes = [];
const stats = { sent: 0, recv: 0, rtt: null, counts: { idle: 0, nonidle: 0, speak: 0, blank: 0 } };
const timeline = []; // recent states, newest last
const TIMELINE_MAX = 300;
let turnCount = 0;

/* --------------------------- helpers --------------------------- */
function log(msg, cls = '') {
  const t = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  const line = document.createElement('div');
  const ts = document.createElement('span');
  ts.className = 't';
  ts.textContent = t;
  const body = document.createElement('span');
  body.className = cls;
  body.textContent = ` ${msg}`;
  line.append(ts, body);
  el.log.appendChild(line);
  while (el.log.childElementCount > 300) el.log.removeChild(el.log.firstElementChild);
  el.log.scrollTop = el.log.scrollHeight;
}

function setConn(text, mode) {
  el.connText.textContent = text;
  if (mode) el.connPill.dataset.on = mode;
  else delete el.connPill.dataset.on;
}

/** Resolve initial ws url: ?ws= param > localStorage > same-host guess > default. */
function initialWsUrl() {
  const q = new URLSearchParams(location.search).get('ws');
  if (q) return q;

  const saved = localStorage.getItem(LS_KEY);
  if (saved) return saved;

  // If this page happens to be served from the inference host itself, prefer it.
  if (location.protocol.startsWith('http') && location.hostname) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (location.port === '8000') return `${proto}//${location.host}/turn`;
    return `${proto}//${location.hostname}:8000/turn`;
  }
  return DEFAULT_WS;
}

function newSessionId() {
  const rnd = Math.random().toString(36).slice(2, 8);
  return `web-${Date.now().toString(36)}-${rnd}`;
}

function float32ToBase64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let bin = '';
  const STEP = 0x8000;
  for (let i = 0; i < bytes.length; i += STEP) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + STEP));
  }
  return btoa(bin);
}

/** Streaming linear-interpolation resampler (native rate -> 16 kHz). */
class Resampler {
  constructor(inRate, outRate) {
    this.ratio = inRate / outRate;
    this.pos = 0;
    this.tail = new Float32Array(0);
  }
  process(input) {
    if (this.ratio === 1) return input;

    const buf = new Float32Array(this.tail.length + input.length);
    buf.set(this.tail, 0);
    buf.set(input, this.tail.length);

    const out = new Float32Array(Math.ceil(buf.length / this.ratio) + 2);
    let p = this.pos;
    let n = 0;
    while (Math.floor(p) + 1 < buf.length) {
      const i = Math.floor(p);
      const f = p - i;
      out[n++] = buf[i] * (1 - f) + buf[i + 1] * f;
      p += this.ratio;
    }
    const keep = Math.floor(p);
    this.tail = buf.slice(keep);
    this.pos = p - keep;
    return out.subarray(0, n);
  }
}

/* --------------------------- websocket --------------------------- */
function connectWs() {
  return new Promise((resolve, reject) => {
    const url = el.serverUrl.value.trim() || DEFAULT_WS;
    localStorage.setItem(LS_KEY, url);
    setConn('连接中…', null);

    let sock;
    try {
      sock = new WebSocket(url);
    } catch (e) {
      reject(e);
      return;
    }

    let settled = false;

    sock.onopen = () => {
      settled = true;
      ws = sock;
      setConn('已连接', '1');
      log(`WebSocket 已连接: ${url}`, 'ok');
      resolve(sock);
    };
    sock.onmessage = (ev) => onServerMessage(ev.data);
    sock.onerror = () => {
      setConn('连接错误', 'err');
      log('WebSocket 错误：请确认推理服务已启动（bash run.sh），且地址/端口正确', 'err');
    };
    sock.onclose = () => {
      if (ws === sock) ws = null;
      if (!settled) {
        settled = true;
        reject(new Error('连接被关闭'));
        return;
      }
      setConn('已断开', running ? 'err' : null);
      log('WebSocket 已关闭', 'warn');
      if (running) stop();
    };

    setTimeout(() => {
      if (!settled) {
        settled = true;
        try { sock.close(); } catch (e) {}
        reject(new Error('连接超时'));
      }
    }, 8000);
  });
}

function onServerMessage(raw) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    log(`无法解析服务端消息: ${raw}`, 'warn');
    return;
  }
  if (data.type !== 'turn_state') return;

  stats.recv += 1;
  const t0 = sendTimes.shift();
  if (t0 !== undefined) {
    const rtt = performance.now() - t0;
    stats.rtt = stats.rtt === null ? rtt : stats.rtt * 0.8 + rtt * 0.2;
  }

  const s = data.state || {};
  applyState(s.state || 'blank', s);
  refreshStats();
}

/* --------------------------- state rendering --------------------------- */
function applyState(name, payload) {
  const meta = STATE_META[name] || { zh: name, en: name, desc: '' };
  el.stateCard.dataset.state = name;
  el.stateName.textContent = meta.zh;
  el.stateEn.textContent = meta.en;
  el.stateDesc.textContent = meta.desc;

  if (stats.counts[name] === undefined) stats.counts[name] = 0;
  stats.counts[name] += 1;

  timeline.push(name);
  while (timeline.length > TIMELINE_MAX) timeline.shift();
  drawTimeline();

  if (name === 'nonidle' || name === 'speak') {
    el.asrSegment.textContent = payload.asr_segment || '—';
    el.asrBuffer.textContent = payload.asr_buffer || '—';
  } else if (name === 'idle') {
    el.asrSegment.textContent = '—';
  }

  if (name === 'speak') {
    addTurn(payload.text || payload.asr_buffer || '(空)');
    el.asrBuffer.textContent = '—';
    el.asrSegment.textContent = '—';
  }
}

function addTurn(text) {
  turnCount += 1;
  const empty = el.turnList.querySelector('.empty');
  if (empty) empty.remove();

  const li = document.createElement('li');
  const idx = document.createElement('span');
  idx.className = 'idx';
  idx.textContent = `#${turnCount}`;
  const span = document.createElement('span');
  span.textContent = text;
  li.append(idx, span);
  el.turnList.appendChild(li);
  el.turnList.scrollTop = el.turnList.scrollHeight;
  el.turnTag.textContent = `${turnCount} 轮`;
  log(`[speak] 整轮转写: ${text}`, 'ok');
}

function refreshStats() {
  el.stSent.textContent = stats.sent;
  el.stRecv.textContent = stats.recv;
  el.stRtt.textContent = stats.rtt === null ? '-- ms' : `${stats.rtt.toFixed(0)} ms`;
  const total = stats.counts.idle + stats.counts.nonidle + stats.counts.speak;
  el.stIdle.textContent = total ? `${((stats.counts.idle / total) * 100).toFixed(0)}%` : '--';
}

/* --------------------------- canvas drawing --------------------------- */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 100;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const waveBuf = new Float32Array(2048);

function drawWave() {
  const { ctx, w, h } = fitCanvas(el.wave);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  if (!analyser) return;
  analyser.getFloatTimeDomainData(waveBuf);

  let sum = 0;
  for (let i = 0; i < waveBuf.length; i++) sum += waveBuf[i] * waveBuf[i];
  const rms = Math.sqrt(sum / waveBuf.length);
  const db = rms > 1e-8 ? 20 * Math.log10(rms) : -100;
  el.meterFill.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
  el.rmsText.textContent = db <= -99 ? '-∞ dB' : `${db.toFixed(1)} dB`;

  const cur = el.stateCard.dataset.state;
  ctx.strokeStyle =
    cur === 'speak' ? '#22c55e' : cur === 'nonidle' ? '#3b82f6' : 'rgba(148,163,196,0.75)';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  const step = waveBuf.length / w;
  for (let x = 0; x < w; x++) {
    const y = h / 2 - waveBuf[Math.floor(x * step)] * (h / 2) * 0.92;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawTimeline() {
  const { ctx, w, h } = fitCanvas(el.timeline);
  ctx.clearRect(0, 0, w, h);

  const colors = { idle: '#64748b', nonidle: '#3b82f6', speak: '#22c55e', blank: '#334155' };
  const bw = w / TIMELINE_MAX;
  const offset = TIMELINE_MAX - timeline.length;

  for (let i = 0; i < timeline.length; i++) {
    const s = timeline[i];
    const x = (offset + i) * bw;
    const isSpeak = s === 'speak';
    if (isSpeak) {
      ctx.fillStyle = 'rgba(34,197,94,0.18)';
      ctx.fillRect(x - bw, 0, bw * 3, h);
    }
    const barH = s === 'nonidle' ? h * 0.7 : isSpeak ? h : h * 0.22;
    ctx.fillStyle = colors[s] || '#334155';
    ctx.fillRect(x, (h - barH) / 2, Math.max(1, bw - 0.4), barH);
  }

  ctx.fillStyle = 'rgba(230,236,255,0.35)';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillText('旧', 4, h - 5);
  ctx.fillText('新', w - 18, h - 5);
}

function loop() {
  drawWave();
  if (running) el.stDur.textContent = `${((performance.now() - startTs) / 1000).toFixed(1)} s`;
  requestAnimationFrame(loop);
}

/* --------------------------- audio capture --------------------------- */
function handlePcm(block) {
  if (!running) return;

  const res = resampler ? resampler.process(block) : block;
  if (res.length === 0) return;

  const merged = new Float32Array(pending.length + res.length);
  merged.set(pending, 0);
  merged.set(res, pending.length);
  pending = merged;

  while (pending.length >= CHUNK) {
    sendChunk(pending.slice(0, CHUNK));
    pending = pending.slice(CHUNK);
  }
}

function sendChunk(chunk) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(
      JSON.stringify({
        type: 'audio',
        session_id: el.sessionId.value.trim(),
        audio: float32ToBase64(chunk),
      })
    );
    sendTimes.push(performance.now());
    if (sendTimes.length > 64) sendTimes.shift();
    stats.sent += 1;
    el.stSent.textContent = stats.sent;
  } catch (e) {
    log(`发送失败: ${e.message}`, 'err');
  }
}

async function listMics() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter((d) => d.kind === 'audioinput');
    const prev = el.micSelect.value;
    el.micSelect.innerHTML = '';
    const auto = document.createElement('option');
    auto.value = '';
    auto.textContent = '默认麦克风';
    el.micSelect.appendChild(auto);
    mics.forEach((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId;
      o.textContent = d.label || `麦克风 ${i + 1}`;
      el.micSelect.appendChild(o);
    });
    if (prev) el.micSelect.value = prev;
  } catch (e) {
    /* ignore */
  }
}

async function start() {
  if (running) return;
  el.btnStart.disabled = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    log('浏览器不支持 getUserMedia。请通过 http://localhost 或 HTTPS 打开本页（不要用 file://）。', 'err');
    el.btnStart.disabled = false;
    return;
  }

  try {
    await connectWs();
  } catch (e) {
    log(`无法连接推理服务: ${e.message}`, 'err');
    setConn('连接失败', 'err');
    el.btnStart.disabled = false;
    return;
  }

  try {
    const deviceId = el.micSelect.value;
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  } catch (e) {
    log(`麦克风授权失败: ${e.name} ${e.message}`, 'err');
    el.btnStart.disabled = false;
    if (ws) ws.close();
    return;
  }

  await listMics();

  const AC = window.AudioContext || window.webkitAudioContext;
  try {
    audioCtx = new AC({ sampleRate: TARGET_SR });
  } catch (e) {
    audioCtx = new AC();
  }
  await audioCtx.resume();

  const nativeSr = audioCtx.sampleRate;
  resampler = nativeSr === TARGET_SR ? null : new Resampler(nativeSr, TARGET_SR);
  el.rateTag.textContent = resampler ? `${nativeSr} Hz → 16000 Hz` : '16000 Hz';
  el.stRate.textContent = `${nativeSr}`;
  log(`AudioContext 采样率 ${nativeSr} Hz${resampler ? '（前端重采样到 16 kHz）' : ''}`);

  const source = audioCtx.createMediaStreamSource(mediaStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.4;
  source.connect(analyser);

  pending = new Float32Array(0);
  sendTimes.length = 0;

  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  mute.connect(audioCtx.destination);

  let ok = false;
  if (audioCtx.audioWorklet) {
    try {
      await audioCtx.audioWorklet.addModule('pcm-worklet.js');
      workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
      });
      workletNode.port.onmessage = (ev) => handlePcm(ev.data);
      source.connect(workletNode);
      workletNode.connect(mute);
      ok = true;
    } catch (e) {
      log(`AudioWorklet 不可用，回退 ScriptProcessor: ${e.message}`, 'warn');
    }
  }

  if (!ok) {
    scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
    scriptNode.onaudioprocess = (ev) =>
      handlePcm(new Float32Array(ev.inputBuffer.getChannelData(0)));
    source.connect(scriptNode);
    scriptNode.connect(mute);
  }

  running = true;
  startTs = performance.now();
  el.btnStop.disabled = false;
  el.serverUrl.disabled = true;
  el.sessionId.disabled = true;
  applyState('blank', {});
  log(`开始推流，session=${el.sessionId.value.trim()}`, 'ok');
}

function stop() {
  running = false;
  el.btnStart.disabled = false;
  el.btnStop.disabled = true;
  el.serverUrl.disabled = false;
  el.sessionId.disabled = false;

  if (workletNode) {
    try { workletNode.port.onmessage = null; workletNode.disconnect(); } catch (e) {}
    workletNode = null;
  }
  if (scriptNode) {
    try { scriptNode.onaudioprocess = null; scriptNode.disconnect(); } catch (e) {}
    scriptNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  analyser = null;
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();

  el.stateCard.dataset.state = 'none';
  el.stateName.textContent = '--';
  el.stateEn.textContent = '已停止';
  el.stateDesc.textContent = '推流已停止。点击「开始」可重新采集麦克风。';
  el.meterFill.style.width = '0%';
  log('已停止', 'warn');
}

function resetSession() {
  if (running) stop();
  el.sessionId.value = newSessionId();
  timeline.length = 0;
  turnCount = 0;
  stats.sent = 0;
  stats.recv = 0;
  stats.rtt = null;
  stats.counts = { idle: 0, nonidle: 0, speak: 0, blank: 0 };
  el.turnList.innerHTML = '<li class="empty">尚无完整轮次（状态变为 <b>speak</b> 时产出）</li>';
  el.turnTag.textContent = '0 轮';
  el.asrSegment.textContent = '—';
  el.asrBuffer.textContent = '—';
  el.stDur.textContent = '0.0 s';
  refreshStats();
  drawTimeline();
  log(`新会话: ${el.sessionId.value}`);
}

/* --------------------------- init --------------------------- */
el.serverUrl.value = initialWsUrl();
el.sessionId.value = newSessionId();
el.btnStart.addEventListener('click', start);
el.btnStop.addEventListener('click', stop);
el.btnReset.addEventListener('click', resetSession);
el.btnClearLog.addEventListener('click', () => (el.log.innerHTML = ''));
window.addEventListener('resize', drawTimeline);
window.addEventListener('beforeunload', () => { if (running) stop(); });

if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener?.('devicechange', listMics);
  listMics();
}

drawTimeline();
loop();
log('页面就绪。请先启动推理服务（bash run.sh），再点击「开始」。');
