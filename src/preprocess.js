import { INPUT_SIZE } from './constants.js';

export function preprocessToTensor(videoEl, lastBox, useNCHW, overlay){
  const W = INPUT_SIZE, H = INPUT_SIZE;
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  const ctx = cvs.getContext('2d', { willReadFrequently: true });

  if (lastBox && overlay){
    const dpr = devicePixelRatio;
    const sx = lastBox.x / dpr, sy = lastBox.y / dpr;
    const sw = lastBox.w / dpr, sh = lastBox.h / dpr;
    ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, W, H);
  } else {
    ctx.drawImage(videoEl, 0, 0, W, H);
  }

  const { data } = ctx.getImageData(0,0,W,H);
  const wh = W*H;

  if (useNCHW) {
    const floats = new Float32Array(wh*3);
    for (let i=0,p=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; floats[i]=(r/127.5-1); floats[i+wh]=(g/127.5-1); floats[i+2*wh]=(b/127.5-1); }
    return new ort.Tensor('float32', floats, [1,3,H,W]);
  } else {
    const floats = new Float32Array(wh*3);
    for (let i=0,p=0,q=0;i<wh;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; floats[q++]=(r/127.5-1); floats[q++]=(g/127.5-1); floats[q++]=(b/127.5-1); }
    return new ort.Tensor('float32', floats, [1,H,W,3]);
  }
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
