import { INPUT_SIZE, LOOP_INTERVAL_MS, DEFAULT_CROP_MODE, DEFAULT_NORM_MODE, DEFAULT_ROI_PAD, DEFAULT_BOX_EMA } from './constants.js';
import { $, setText, setStatus, logDiag, resizeOverlay } from './ui.js';
import { startCamera, stopCamera, hasStream } from './camera.js';
import { loadModel } from './model.js';
import { findTabletBox, drawBox } from './box.js';
import { preprocessToTensor, topkSoftmax } from './preprocess.js';

let state = {
  session: null,
  INPUT_NAME: null,
  OUTPUT_NAME: null,
  useNCHW: false,
  mapping: null,
  usingBack: true,
  
  // Настройки препроцессинга (управляются через UI)
  cropMode: DEFAULT_CROP_MODE,
  normMode: DEFAULT_NORM_MODE,
  roiPad: DEFAULT_ROI_PAD,
  boxEma: DEFAULT_BOX_EMA,
  
  lastBox: null,
  rafId: null,
  lastTs: 0,
  video: null,
  overlay: null,
  octx: null,
};

async function onStartClick(){
  try{
    $('#btnStart').disabled = true;
    setStatus('Подготовка…');
    const model = await loadModel();
    Object.assign(state, model);

    state.video = await startCamera(state.usingBack);
    state.overlay = document.getElementById('overlay');
    state.octx = state.overlay.getContext('2d');

    resizeOverlay(state.video, state.overlay);
    window.addEventListener('resize', onResize, { passive:true });

    $('#btnStop').disabled = false;
    setStatus('Камера запущена. Распознаём…', null);
    state.rafId = requestAnimationFrame(loop);
  }catch(e){
    $('#btnStart').disabled = false;
    setText('result','Ошибка запуска');
    setText('sub', e?.message || String(e));
    logDiag(e?.stack || String(e));
  }
}

function onResize(){ resizeOverlay(state.video, state.overlay); }

async function onStopClick(){
  if (state.rafId) cancelAnimationFrame(state.rafId);
  await stopCamera();
  window.removeEventListener('resize', onResize);
  $('#btnStop').disabled = true;
  $('#btnStart').disabled = false;
  setStatus('Остановлено.', null);
  drawBox(state.octx, state.overlay, null);
}

async function onFlipClick(){
  state.usingBack = !state.usingBack;
  if (hasStream()){
    state.video = await startCamera(state.usingBack);
    resizeOverlay(state.video, state.overlay);
  }
}

function loop(ts){
  if (!state.lastTs || ts - state.lastTs >= LOOP_INTERVAL_MS) {
    state.lastTs = ts;
    try{
      // Поиск рамки (если режим auto)
      if (state.cropMode === 'auto') {
        state.lastBox = findTabletBox(state.video, state.overlay, state.boxEma, state.lastBox) || state.lastBox;
        drawBox(state.octx, state.overlay, state.lastBox);
      } else {
        drawBox(state.octx, state.overlay, null);
      }

      // Препроцесс + инференс
      const x = preprocessToTensor(
        state.video, 
        state.cropMode, 
        state.normMode, 
        state.roiPad, 
        state.lastBox, 
        state.useNCHW, 
        state.overlay
      );
      
      state.session.run({ [state.INPUT_NAME]: x }).then(out => {
        const logits = out[state.OUTPUT_NAME].data;
        const { top1Idx, top1Prob, top3 } = topkSoftmax(logits, 3);
        const name = state.mapping[String(top1Idx)] ?? `class_${top1Idx}`;
        
        setText('result', `🧩 ${name} (${(top1Prob*100).toFixed(1)}%)`);
        
        const info = [
          `Класс #${top1Idx}`,
          `Формат: ${state.useNCHW?'NCHW':'NHWC'}`,
          `Crop: ${state.cropMode}`,
          `Norm: ${state.normMode}`
        ].join(' • ');
        setText('sub', info);
        
        document.getElementById('topk').innerHTML = top3.map(
          ({idx, prob}) => `<span class="pill">${state.mapping[String(idx)] ?? 'class_'+idx} — ${(prob*100).toFixed(1)}%</span>`
        ).join('');
      }).catch(e=>{
        setText('result','Ошибка инференса');
        setText('sub', e?.message || String(e));
        logDiag(e?.stack || String(e));
      });
    }catch(e){
      setText('result','Ошибка цикла');
      setText('sub', e?.message || String(e));
      logDiag(e?.stack || String(e));
    }
  }
  state.rafId = requestAnimationFrame(loop);
}

// ========== UI События ==========
document.getElementById('btnStart').addEventListener('click', onStartClick);
document.getElementById('btnStop').addEventListener('click', onStopClick);
document.getElementById('btnFlip').addEventListener('click', onFlipClick);

// Режим кадрирования
document.getElementById('selCrop').addEventListener('change', (e)=>{
  state.cropMode = e.target.value;
  if(state.cropMode !== 'auto') {
    state.lastBox = null;
    drawBox(state.octx, state.overlay, null);
  }
  logDiag(`Crop mode: ${state.cropMode}`);
});

// Режим нормализации
document.getElementById('selNorm').addEventListener('change', (e)=>{
  state.normMode = e.target.value;
  logDiag(`Norm mode: ${state.normMode}`);
});

// ROI padding слайдер
document.getElementById('rngPad').addEventListener('input', (e)=>{
  state.roiPad = parseFloat(e.target.value);
  document.getElementById('valPad').textContent = state.roiPad.toFixed(2);
});

// EMA сглаживание слайдер
document.getElementById('rngEma').addEventListener('input', (e)=>{
  state.boxEma = parseFloat(e.target.value);
  document.getElementById('valEma').textContent = state.boxEma.toFixed(2);
});

// Начальный подсказочный текст
setText('sub', `Вход: ${INPUT_SIZE}×${INPUT_SIZE}. Нажми «Старт камеры».`);
logDiag(`Defaults: crop=${state.cropMode}, norm=${state.normMode}, pad=${state.roiPad}, ema=${state.boxEma}`);
