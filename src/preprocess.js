import { INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD } from './constants.js';

// Препроцессинг с поддержкой разных режимов кадрирования и нормализации
export function preprocessToTensor(videoEl, cropMode, normMode, roiPad, lastBox, useNCHW, overlay){
  const W = INPUT_SIZE, H = INPUT_SIZE;
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  const ctx = cvs.getContext('2d', { willReadFrequently: true });

  // ========== КАДРИРОВАНИЕ ==========
  if (cropMode === 'full') {
    // Full frame: весь кадр
    ctx.drawImage(videoEl, 0, 0, W, H);
    
  } else if (cropMode === 'center') {
    // Center-crop: квадрат из центра видео
    const vw = videoEl.videoWidth || videoEl.width;
    const vh = videoEl.videoHeight || videoEl.height;
    const size = Math.min(vw, vh);
    const sx = (vw - size) / 2;
    const sy = (vh - size) / 2;
    ctx.drawImage(videoEl, sx, sy, size, size, 0, 0, W, H);
    
  } else if (cropMode === 'auto' && lastBox && overlay) {
    // Auto-ROI: детектор рамки с паддингом
    const dpr = devicePixelRatio;
    let sx = lastBox.x / dpr;
    let sy = lastBox.y / dpr;
    let sw = lastBox.w / dpr;
    let sh = lastBox.h / dpr;
    
    // Применяем паддинг
    if (roiPad > 0) {
      const padX = sw * roiPad;
      const padY = sh * roiPad;
      sx = Math.max(0, sx - padX);
      sy = Math.max(0, sy - padY);
      sw = sw + 2*padX;
      sh = sh + 2*padY;
    }
    
    ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, W, H);
  } else {
    // Fallback: full frame
    ctx.drawImage(videoEl, 0, 0, W, H);
  }

  const { data } = ctx.getImageData(0,0,W,H);
  const wh = W*H;

  // ========== НОРМАЛИЗАЦИЯ ==========
  const floats = new Float32Array(wh*3);
  
  if (normMode === 'zero1') {
    // [0, 1]: x/255
    if (useNCHW) {
      for (let i=0,p=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[i]=(r/255); floats[i+wh]=(g/255); floats[i+2*wh]=(b/255); }
    } else {
      for (let i=0,p=0,q=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[q++]=(r/255); floats[q++]=(g/255); floats[q++]=(b/255); }
    }
    
  } else if (normMode === 'imagenet') {
    // ImageNet: (x/255 - mean)/std
    const [mr, mg, mb] = IMAGENET_MEAN;
    const [sr, sg, sb] = IMAGENET_STD;
    if (useNCHW) {
      for (let i=0,p=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[i]=((r/255-mr)/sr); floats[i+wh]=((g/255-mg)/sg); floats[i+2*wh]=((b/255-mb)/sb); }
    } else {
      for (let i=0,p=0,q=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[q++]=((r/255-mr)/sr); floats[q++]=((g/255-mg)/sg); floats[q++]=((b/255-mb)/sb); }
    }
    
  } else {
    // neg1to1 (default): x/127.5 - 1
    if (useNCHW) {
      for (let i=0,p=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[i]=(r/127.5-1); floats[i+wh]=(g/127.5-1); floats[i+2*wh]=(b/127.5-1); }
    } else {
      for (let i=0,p=0,q=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; 
        floats[q++]=(r/127.5-1); floats[q++]=(g/127.5-1); floats[q++]=(b/127.5-1); }
    }
  }

  return useNCHW 
    ? new ort.Tensor('float32', floats, [1,3,H,W])
    : new ort.Tensor('float32', floats, [1,H,W,3]);
}

export function topkSoftmax(logits, k=3){
  let maxL=-Infinity; for (let i=0;i<logits.length;i++) if (logits[i]>maxL) maxL=logits[i];
  let sum=0; const exps=new Float32Array(logits.length);
  for (let i=0;i<logits.length;i++){ const e=Math.exp(logits[i]-maxL); exps[i]=e; sum+=e; }
  const probs=new Float32Array(logits.length);
  for (let i=0;i<logits.length;i++) probs[i]=exps[i]/sum;
  const idxs=[...probs.keys()].sort((a,b)=>probs[b]-probs[a]);
  return { top1Idx: idxs[0], top1Prob: probs[idxs[0]], top3: idxs.slice(0,k).map(i=>({idx:i, prob:probs[i]})) };
}
