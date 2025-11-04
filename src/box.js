import { EDGE_SIZE } from './constants.js';

// Поиск рамки планшета на уменьшенном кадре (Sobel + интегральное изображение)
export function findTabletBox(videoEl, overlay){
  if (!overlay) return null;
  const vw = overlay.width, vh = overlay.height;
  const s = EDGE_SIZE;
  const cvs = document.createElement('canvas');
  cvs.width = s;
  cvs.height = Math.max(1, Math.floor(s * (vh / Math.max(1, vw))));
  const ctx = cvs.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, cvs.width, cvs.height);
  const { data, width:W, height:H } = ctx.getImageData(0,0,cvs.width,cvs.height);

  const gxK = [-1,0,1,-2,0,2,-1,0,1];
  const gyK = [-1,-2,-1,0,0,0,1,2,1];
  const gray = new Float32Array(W*H);
  for(let i=0,p=0;i<gray.length;i++){ const r=data[p++],g=data[p++],b=data[p++]; p++; gray[i]=(0.299*r+0.587*g+0.114*b); }
  const mag = new Float32Array(W*H);
  for(let y=1;y<H-1;y++){
    for(let x=1;x<W-1;x++){
      let gx=0, gy=0, k=0;
      for(let j=-1;j<=1;j++){
        for(let i=-1;i<=1;i++){
          const v = gray[(y+j)*W + (x+i)];
          gx += v * gxK[k]; gy += v * gyK[k]; k++;
        }
      }
      mag[y*W + x] = Math.hypot(gx,gy);
    }
  }
  // интегральная сумма
  const ii = new Float32Array((W+1)*(H+1));
  for(let y=1;y<=H;y++){
    let row=0;
    for(let x=1;x<=W;x++){
      row += mag[(y-1)*W + (x-1)];
      ii[y*(W+1)+x] = ii[(y-1)*(W+1)+x] + row;
    }
  }
  const sumRect = (x0,y0,x1,y1)=> ii[y1*(W+1)+x1] - ii[y0*(W+1)+x1] - ii[y1*(W+1)+x0] + ii[y0*(W+1)+x0];

  const aspects = [1.33, 1.5, 1.6];
  let best = {score:-1, x:0,y:0,w:W,h:H};
  for(const ar of aspects){
    for(let w = Math.floor(W*0.45); w <= Math.floor(W*0.95); w += Math.max(8, Math.floor(W*0.08))){
      const h = Math.floor(w / ar);
      if (h<16 || h>H) continue;
      for(let x=1; x+w < W; x += Math.max(6, Math.floor(W*0.06))){
        for(let y=1; y+h < H; y += Math.max(6, Math.floor(H*0.06))){
          const s = sumRect(x, y, x+w, y+h) / (w*h);
          if (s > best.score) best = {score:s, x, y, w, h};
        }
      }
    }
  }
  const scaleX = vw / W, scaleY = vh / H;
  return {
    x: Math.max(0, Math.floor(best.x * scaleX)),
    y: Math.max(0, Math.floor(best.y * scaleY)),
    w: Math.max(1, Math.floor(best.w * scaleX)),
    h: Math.max(1, Math.floor(best.h * scaleY)),
  };
}

export function drawBox(ctx, overlay, box){
  if (!ctx || !overlay) return;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if (!box) return;
  ctx.lineWidth = 3 * devicePixelRatio;
  ctx.strokeStyle = 'rgba(80,160,255,0.95)';
  ctx.setLineDash([10*devicePixelRatio, 6*devicePixelRatio]);
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.setLineDash([]);
}
