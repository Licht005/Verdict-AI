/* ── Verdict AI — app.js ── */

const API = 'https://lucaslicht-verdict-ai.hf.space';
let mode = 'text';
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentAudio = null;

function setMode(m) {
  mode = m;
  document.getElementById('textMode').classList.toggle('hidden', m !== 'text');
  document.getElementById('voiceMode').classList.toggle('hidden', m !== 'voice');
  document.getElementById('textModeBtn').classList.toggle('active', m === 'text');
  document.getElementById('voiceModeBtn').classList.toggle('active', m === 'voice');
  document.getElementById('modeLabel').textContent =
    m === 'text' ? 'Text mode · Ask any question' : 'Voice mode · Speak your question';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function fill(q) {
  const input = document.getElementById('questionInput');
  input.value = q;
  input.focus();
  autoResize(input);
}

async function sendText() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('textError').innerHTML = '';
  hideEmpty();
  addMsg('messages', 'user', question);
  const thinking = addThinking('messages');

  try {
    const res = await fetch(`${API}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    thinking.remove();
    addMsg('messages', 'assistant', data.answer);
  } catch (e) {
    thinking.remove();
    document.getElementById('textError').innerHTML =
      `<div class="error-msg">⚠ ${e.message} — is the server running? (python api.py)</div>`;
  }
  document.getElementById('sendBtn').disabled = false;
}

async function toggleRecording() {
  if (isRecording) stopRecording();
  else await startRecording();
}

async function startRecording() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = processVoice;
    mediaRecorder.start();
    isRecording = true;
    setOrb('recording');
  } catch {
    document.getElementById('voiceError').innerHTML =
      `<div class="error-msg">⚠ Microphone access denied.</div>`;
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    isRecording = false;
    setOrb('processing');
  }
}

async function processVoice() {
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const fd = new FormData();
  fd.append('audio', blob, 'recording.webm');
  document.getElementById('voiceError').innerHTML = '';

  try {
    const res = await fetch(`${API}/ask-voice`, { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    const question = data.question ? atob(data.question) : '(voice question)';
    const answer   = data.answer   ? atob(data.answer)   : '';

    addMsg('voiceMessages', 'user', question);
    addMsg('voiceMessages', 'assistant', answer + (answer.length >= 800 ? '…' : ''));

    if (data.tts_fallback || !data.audio) {
      // ElevenLabs unavailable — fall back to browser speechSynthesis
      showToast('Using browser voice (ElevenLabs quota reached)');
      setOrb('playing');
      const utterance = new SpeechSynthesisUtterance(answer);
      utterance.rate = 0.95;
      utterance.onend = () => {
        setOrb('idle');
        document.getElementById('stopAudioBtn').classList.remove('visible');
      };
      document.getElementById('stopAudioBtn').classList.add('visible');
      currentAudio = { pause: () => speechSynthesis.cancel(), currentTime: 0 };
      speechSynthesis.speak(utterance);
    } else {
      // Decode base64 audio and play it
      const audioBytes = atob(data.audio);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }
      const audioBlob = new Blob([audioArray], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(audioBlob);
      currentAudio = new Audio(url);
      setOrb('playing');
      document.getElementById('stopAudioBtn').classList.add('visible');
      currentAudio.play();
      currentAudio.onended = () => {
        setOrb('idle');
        document.getElementById('stopAudioBtn').classList.remove('visible');
        URL.revokeObjectURL(url);
      };
    }
  } catch (e) {
    setOrb('idle');
    document.getElementById('voiceError').innerHTML =
      `<div class="error-msg">⚠ ${e.message}</div>`;
  }
}

function setOrb(state) {
  const btn    = document.getElementById('voiceBtn');
  const status = document.getElementById('voiceStatus');
  const hint   = document.getElementById('voiceHint');
  const wave   = document.getElementById('waveform');
  const icon   = document.getElementById('orbIcon');

  btn.className = 'voice-btn';
  wave.className = 'waveform';

  const map = {
    idle:       { icon:'🎙', s:'Press to speak',            h:'Tap to begin recording' },
    recording:  { icon:'⏹',  s:'Recording…',                h:'Tap again to stop',         cls:'recording', wave:true },
    processing: { icon:'⟳',  s:'Transcribing & analysing…', h:'Please wait',               cls:'processing' },
    playing:    { icon:'🔊', s:'Playing response…',          h:'Audio playing',             cls:'playing' },
  };

  const s = map[state] || map.idle;
  icon.textContent   = s.icon;
  status.textContent = s.s;
  hint.textContent   = s.h;
  if (s.cls)  btn.classList.add(s.cls);
  if (s.wave) wave.classList.add('active');
}

function clearChat() {
  const container = document.getElementById('messages');
  container.innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">📜</div>
      <div class="empty-title">Ask about Ghana's Constitution</div>
      <div class="empty-sub">Query any article, right, chapter or provision. Verdict AI cites the exact clause.</div>
    </div>`;
}

function clearVoiceChat() {
  document.getElementById('voiceMessages').innerHTML = '';
}

function stopAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  if (isRecording) stopRecording();
  setOrb('idle');
  document.getElementById('stopAudioBtn').classList.remove('visible');
}

function hideEmpty() {
  const e = document.getElementById('emptyState');
  if (e) e.remove();
}

function addMsg(containerId, role, text) {
  const c = document.getElementById(containerId);
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'YOU' : 'AI'}</div>
    <div class="msg-body">
      <div class="msg-label">${role === 'user' ? 'Your question' : 'Verdict AI'}</div>
      <div class="msg-text">${esc(text)}</div>
    </div>`;
  c.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

function addThinking(containerId) {
  const c = document.getElementById(containerId);
  const el = document.createElement('div');
  el.className = 'thinking';
  el.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Consulting the Constitution…</span>`;
  c.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

function showToast(msg) {
  const existing = document.getElementById('verdict-toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.id = 'verdict-toast';
  toast.className = 'toast';
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.classList.add('toast-visible'), 10);
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 400);
  }, 4000);
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br/>');
}
