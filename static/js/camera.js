/**
 * camera.js — Shared webcam capture + /api/predict polling
 * ==========================================================
 * Used by both practice.html and quiz.html.
 *
 * Usage
 * -----
 *   const cam = new BIMCamera({
 *     videoEl:        document.getElementById('video'),
 *     canvasEl:       document.getElementById('canvas'),
 *     onResult:       (result) => { ... },   // called each time API returns
 *     onStateChange:  (state) => { ... },    // 'starting' | 'live' | 'stopped' | 'error'
 *     intervalMs:     300,                   // ms between API calls
 *     smoothWindow:   7,                     // majority-vote history length
 *   });
 *
 *   cam.start();   // request camera + begin polling
 *   cam.stop();    // stop camera + polling
 *   cam.reset();   // clear smoothing history only (keep camera running)
 *
 * result shape (from /api/predict):
 *   { status, label, confidence, all_scores, bbox }
 *   status: "ok" | "no_hand" | "low_confidence" | "error" | "decode_error"
 */

class BIMCamera {
  constructor(opts = {}) {
    this.videoEl       = opts.videoEl;
    this.canvasEl      = opts.canvasEl || document.createElement('canvas');
    this.onResult      = opts.onResult      || (() => {});
    this.onStateChange = opts.onStateChange || (() => {});
    this.intervalMs    = opts.intervalMs    ?? 300;
    this.smoothWindow  = opts.smoothWindow  ?? 7;

    this._stream    = null;
    this._timer     = null;
    this._history   = [];
    this._ctx       = this.canvasEl.getContext('2d');
    this._running   = false;
  }

  async start() {
    this.onStateChange('starting');
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      this.videoEl.srcObject = this._stream;
      await new Promise(res => this.videoEl.onloadedmetadata = res);
      this._running = true;
      this.onStateChange('live');
      this._timer = setInterval(() => this._captureAndPredict(), this.intervalMs);
    } catch (err) {
      console.error('[BIMCamera] start error:', err);
      this.onStateChange('error');
    }
  }

  stop() {
    clearInterval(this._timer);
    this._running = false;
    if (this._stream) this._stream.getTracks().forEach(t => t.stop());
    this._stream = null;
    this.onStateChange('stopped');
  }

  reset() {
    this._history = [];
  }

  isRunning() { return this._running; }

  // ── Private ──────────────────────────────────────────────────────────────────
  async _captureAndPredict() {
    if (!this._running || this.videoEl.readyState < 2) return;

    const w = this.videoEl.videoWidth;
    const h = this.videoEl.videoHeight;
    this.canvasEl.width  = w;
    this.canvasEl.height = h;

    // Mirror to match the video display (CSS transform: scaleX(-1))
    this._ctx.save();
    this._ctx.scale(-1, 1);
    this._ctx.drawImage(this.videoEl, -w, 0);
    this._ctx.restore();

    this.canvasEl.toBlob(async (blob) => {
      if (!blob || !this._running) return;
      const fd = new FormData();
      fd.append('frame', blob, 'frame.jpg');

      try {
        const res  = await fetch('/api/predict', { method: 'POST', body: fd });
        const data = await res.json();
        this._applySmoothing(data);
      } catch (err) {
        console.warn('[BIMCamera] predict error:', err);
      }
    }, 'image/jpeg', 0.80);
  }

  _applySmoothing(data) {
    if (data.status === 'ok' && data.label) {
      this._history.push(data.label);
      if (this._history.length > this.smoothWindow)
        this._history.shift();
      const smoothed = _mode(this._history);
      // Attach smoothed label without mutating the original data object
      this.onResult({ ...data, smoothed_label: smoothed });
    } else {
      if (data.status === 'no_hand') this._history = [];
      this.onResult({ ...data, smoothed_label: null });
    }
  }
}

// Majority-vote helper
function _mode(arr) {
  if (!arr.length) return null;
  const f = {};
  for (const v of arr) f[v] = (f[v] || 0) + 1;
  return Object.entries(f).sort((a, b) => b[1] - a[1])[0][0];
}
