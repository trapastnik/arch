export const $ = (s) => document.querySelector(s);
export const setText = (id, t) => { const el = document.getElementById(id); if (el) el.innerText = t; };

export function logDiag(obj){
  const el = document.getElementById('diag');
  if (!el) return;
  el.textContent += (typeof obj==='string' ? obj : JSON.stringify(obj,null,2)) + '\n';
}

export function setStatus(txt, pct=null){
  setText('result', txt);
  const wrap = document.getElementById('progressWrap');
  const bar  = document.getElementById('progress');
  if (!wrap || !bar) return;
  if (pct === null) { wrap.style.display = 'none'; }
  else {
    wrap.style.display = 'block';
    bar.style.width = `${Math.max(0, Math.min(100, Math.round(pct*100)))}%`;
  }
}

export function resizeOverlay(videoEl, overlay){
  if (!videoEl || !overlay) return;
  const rect = videoEl.getBoundingClientRect();
  overlay.width  = Math.max(1, Math.floor(rect.width  * devicePixelRatio));
  overlay.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
}
