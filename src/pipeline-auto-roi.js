/* src/pipeline-auto-roi.js — Minimal & Debug UI + Auto-ROI + warp + anti-moire + normalize + ONNX + EMA */

(async function () {
  // ==== CONFIG ===============================================================
  const MODEL_URL   = 'web_model/model.onnx';
  const INPUT_SIZE  = 224;
  const MEAN        = [0.485, 0.456, 0.406];  // как при обучении
  const STD         = [0.229, 0.224, 0.225];
  const TEMP        = 1.5;     // temperature scaling (стабилизирует)
  const EMA_ALPHA   = 0.72;    // сглаживание вероятностей
  const CONF_THRESH = 0.75;
  const MAX_ROIS    = 3;       // показываем до 3 планшетов на кадр

  // Katarsis в модели — индексы 3,4,5 (для фильтрации — опционально)
  const KATARSIS_INDEXES = [3,4,5];

  // Детектор квадратных планшетов (контуры/аппроксимация)
  const DETCFG = { canny1: 65, canny2: 170, eps: 0.02, minAreaRatio: 0.05, maxSideRatio: 1.6 };

  // ==== BOOTSTRAP ============================================================
  await new Promise(res => {
    if (window.cv && cv.getBuildInformation) return res();
    const t = setInterval(() => { if (window.cv && cv.getBuildInformation) { clearInterval(t); res(); } }, 50);
  });

  let id2label = [];
  try {
    const raw = await fetch('class_mapping.json').then(r=>r.json());
    id2label = Array.isArray(raw) ? raw : Object.keys(raw).sort((a,b)=>(+a)-(+b)).map(k=>raw[k]);
  } catch {}

  const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm','webgl'] });

  // ==== DOM HOOKS ============================================================
  const video   = document.querySelector('video') || document.getElementById('camera');
  const overlay = document.getElementById('overlay');
  const hud     = document.getElementById('hud');
  const dbgBtn  = document.getElementById('debugToggle');
  const dbgPane = document.getElementById('debugPanel');
  const dbgMet  = document.getElementById('debugMetrics');
  const roiStrip= document.getElementById('roiStrip');

  if (!video) { console.warn('[arch] <video> not found'); return; }
  const ctxOv = overlay.getContext('2d');

  // ==== UTILS ================================================================
  const state = {
    debug: false,
    ema: new Map(), // key -> {v, ready}
    fps: 0, tPrev: performance.now(),
    timings: { det: 0, warp: 0, infer: 0, total: 0 },
    last: { nQuads: 0, rois: [] },
  };

  function toggleDebug(force) {
    state.debug = (force ?? !state.debug);
    dbgPane.style.display = state.debug ? 'block' : 'none';
  }
  dbgBtn.onclick = () => toggleDebug();
  window.addEventListener('keydown', e => { if (e.key.toLowerCase()==='d') toggleDebug(); });

  function softmaxWithT(logits, T=1) {
    const s = logits.map(v => v / T);
    const m = Math.max(...s);
    const ex = s.map(v => Math.exp(v - m));
    const sum = ex.reduce((a,b)=>a+b,0);
    return ex.map(v => v/sum);
  }
  function emaInit(n){ return { v:new Float32Array(n), ready:false }; }
  function emaUpdate(st, probs, a){
    if(!st.ready){ st.v.set(probs); st.ready=true; }
    else for(let i=0;i<probs.length;i++) st.v[i]=a*st.v[i]+(1-a)*probs[i];
    return st.v;
  }
  function orderQuad(pts) {
    const s = pts.map(p=>p.x+p.y), d = pts.map(p=>p.x-p.y);
    const tl = pts[s.indexOf(Math.min(...s))];
    const br = pts[s.indexOf(Math.max(...s))];
    const tr = pts[d.indexOf(Math.min(...d))];
    const bl = pts[d.indexOf(Math.max(...d))];
    return [tl,tr,br,bl];
  }
  function detectQuads(mat, cfg) {
    const { canny1, canny2, eps, minAreaRatio, maxSideRatio } = cfg;
    const W = mat.cols, H = mat.rows, MIN = minAreaRatio*W*H;

    let gray = new cv.Mat(); cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    cv.medianBlur(gray, gray, 3); cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0);
    let edges = new cv.Mat(); cv.Canny(gray, edges, canny1, canny2);

    let contours = new cv.MatVector(), hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const quads = [];
    for (let i=0;i<contours.size();i++){
      const cnt = contours.get(i);
      const peri = cv.arcLength(cnt,true);
      const approx = new cv.Mat(); cv.approxPolyDP(cnt, approx, eps*peri, true);
      if (approx.rows===4 && cv.isContourConvex(approx)) {
        const area = cv.contourArea(approx);
        if (area >= MIN) {
          const pts = []; for (let j=0;j<4;j++){ const p=approx.intPtr(j); pts.push({x:p[0],y:p[1]}); }
          const sides = [
            Math.hypot(pts[1].x-pts[0].x, pts[1].y-pts[0].y),
            Math.hypot(pts[2].x-pts[1].x, pts[2].y-pts[1].y),
            Math.hypot(pts[3].x-pts[2].x, pts[3].y-pts[2].y),
            Math.hypot(pts[0].x-pts[3].x, pts[0].y-pts[3].y),
          ];
          const ratio = Math.max(...sides)/(Math.min(...sides)+1e-6);
          if (ratio < maxSideRatio) quads.push(pts);
        }
      }
      approx.delete(); cnt.delete();
    }
    gray.delete(); edges.delete(); contours.delete(); hierarchy.delete();

    quads.sort((a,b)=>{
      const A=(qq)=>{const m=cv.matFromArray(qq.length,1,cv.CV_32SC2,qq.flatMap(p=>[p.x,p.y])); const v=cv.contourArea(m); m.delete(); return v;};
      return A(b)-A(a);
    });
    return quads;
  }
  function warpQuadToCanvas(srcCanvas, quad, outSize=256) {
    const [tl,tr,br,bl] = orderQuad(quad);
    const src = cv.imread(srcCanvas), dst = new cv.Mat();
    const srcTri = cv.matFromArray(4,1,cv.CV_32FC2,[tl.x,tl.y, tr.x,tr.y, br.x,br.y, bl.x,bl.y]);
    const dstTri = cv.matFromArray(4,1,cv.CV_32FC2,[0,0, outSize,0, outSize,outSize, 0,outSize]);
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    cv.warpPerspective(src, dst, M, new cv.Size(outSize,outSize), cv.INTER_CUBIC, cv.BORDER_REPLICATE);
    const out = document.createElement('canvas'); out.width=outSize; out.height=outSize; cv.imshow(out,dst);
    src.delete(); dst.delete(); srcTri.delete(); dstTri.delete(); M.delete();
    return out;
  }
  function antiMoiré(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    // downscale -> upscale сглаживает муар
    const t = document.createElement('canvas');
    t.width = Math.max(1, Math.round(canvas.width*0.75));
    t.height= Math.max(1, Math.round(canvas.height*0.75));
    t.getContext('2d').drawImage(canvas,0,0,t.width,t.height);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(t,0,0,canvas.width,canvas.height);
    return canvas;
  }
  function toTensorCHW(canvas, size=INPUT_SIZE) {
    const tmp = document.createElement('canvas'); tmp.width=size; tmp.height=size;
    tmp.getContext('2d').drawImage(canvas, 0,0,size,size);
    const { data } = tmp.getContext('2d').getImageData(0,0,size,size);
    const N = size*size, out = new Float32Array(3*N);
    for (let i=0,p=0;i<data.length;i+=4,p++){
      const r=data[i]/255, g=data[i+1]/255, b=data[i+2]/255;
      out[p]      = (r-MEAN[0])/STD[0];
      out[p+N]    = (g-MEAN[1])/STD[1];
      out[p+2*N]  = (b-MEAN[2])/STD[2];
    }
    return new ort.Tensor('float32', out, [1,3,size,size]);
  }
  function drawQuad(ctx, q, col='rgba(0,255,0,0.9)'){ ctx.beginPath(); ctx.moveTo(q[0].x,q[0].y); for(let i=1;i<4;i++) ctx.lineTo(q[i].x,q[i].y); ctx.closePath(); ctx.strokeStyle=col; ctx.lineWidth=3; ctx.stroke(); }

  // ==== MAIN LOOP ============================================================
  function resizeOverlayToVideo(){
    overlay.width  = video.videoWidth  || video.clientWidth;
    overlay.height = video.videoHeight || video.clientHeight;
  }

  async function tick() {
    const t0 = performance.now();
    if (video.readyState >= 2) {
      resizeOverlayToVideo();
      const W=overlay.width, H=overlay.height;
      const frame = document.createElement('canvas'); frame.width=W; frame.height=H;
      frame.getContext('2d').drawImage(video,0,0,W,H);

      // DETECT
      const m0 = performance.now();
      const srcMat = cv.imread(frame);
      const quads = detectQuads(srcMat, DETCFG);
      srcMat.delete();
      const tDet = performance.now()-m0;

      ctxOv.clearRect(0,0,W,H);
      const rois = [];

      for (let i=0;i<Math.min(quads.length, MAX_ROIS); i++){
        const q = quads[i];
        drawQuad(ctxOv, q, 'rgba(0,255,0,0.9)');

        // WARP + ANTI-MOIRE
        const m1=performance.now();
        let roi = warpQuadToCanvas(frame, q, 256);
        roi = antiMoiré(roi);

        // PREPROCESS -> TENSOR
        const tensor = toTensorCHW(roi, INPUT_SIZE);
        const tWarp = performance.now()-m1;

        // INFER
        const m2=performance.now();
        const out = await session.run({ [session.inputNames[0]]: tensor });
        const logits = Array.from(out[session.outputNames[0]].data);
        const probs  = softmaxWithT(logits, TEMP);
        const tInf = performance.now()-m2;

        // EMA per ROI
        const key = `roi_${i}`;
        if(!state.ema.has(key)) state.ema.set(key, emaInit(probs.length));
        const smooth = emaUpdate(state.ema.get(key), probs, EMA_ALPHA);

        const ranked = Array.from(smooth).map((p,idx)=>({idx,p})).sort((a,b)=>b.p-a.p);
        const top1 = ranked[0];
        const label = id2label[top1.idx] ?? `class_${top1.idx}`;
        const confident = top1.p >= CONF_THRESH;

        // Подпись над левым верхним углом ROI
        ctxOv.fillStyle = confident ? 'rgba(0,128,0,0.85)' : 'rgba(0,0,0,0.55)';
        ctxOv.font = '16px system-ui, sans-serif';
        const txt = `${label} ${(top1.p*100).toFixed(1)}%`;
        ctxOv.fillRect(q[0].x+6, q[0].y-18, ctxOv.measureText(txt).width+8, 20);
        ctxOv.fillStyle = '#fff';
        ctxOv.fillText(txt, q[0].x+10, q[0].y-3);

        rois.push({
          cropEl: roi,
          top1, top3: ranked.slice(0,3).map(r=>({label: id2label[r.idx] ?? `class_${r.idx}`, prob:r.p})),
          tWarp, tInf
        });
      }

      // HUD (верхний левый бейдж)
      const now = performance.now();
      const dt = now - state.tPrev; state.tPrev = now;
      state.fps = 1000/dt;
      state.timings = { det: tDet, warp: rois.reduce((a,r)=>a+r.tWarp,0), infer: rois.reduce((a,r)=>a+r.tInf,0), total: performance.now()-t0 };

      hud.textContent = `ROIs: ${Math.min(quads.length, MAX_ROIS)} | FPS: ${state.fps.toFixed(1)} | det:${tDet|0}ms inf:${state.timings.infer|0}ms`;

      // Debug panel
      if (state.debug) {
        const lines = [];
        lines.push(`FPS: ${state.fps.toFixed(1)}`);
        lines.push(`Detections: ${quads.length} (showing ${Math.min(quads.length, MAX_ROIS)})`);
        lines.push(`Timings: det=${state.timings.det.toFixed(1)}ms, warp=${state.timings.warp.toFixed(1)}ms, infer=${state.timings.infer.toFixed(1)}ms, total=${state.timings.total.toFixed(1)}ms`);
        lines.push(`Params: canny=[${DETCFG.canny1},${DETCFG.canny2}] eps=${DETCFG.eps} minArea=${DETCFG.minAreaRatio} maxSide=${DETCFG.maxSideRatio} | EMA=${EMA_ALPHA} T=${TEMP} thresh=${CONF_THRESH}`);
        rois.forEach((r, i)=>{
          const t3 = r.top3.map(o=>`${o.label} ${(o.prob*100).toFixed(1)}%`).join(' | ');
          lines.push(`ROI#${i}: ${t3}`);
        });
        dbgMet.innerHTML = `<pre>${lines.join('\n')}</pre>`;

        // Мини-превью кропов (после warp+anti-moire)
        roiStrip.innerHTML = '';
        rois.forEach(r=>{
          const c = document.createElement('canvas');
          c.width=80; c.height=80;
          c.getContext('2d').drawImage(r.cropEl,0,0,80,80);
          roiStrip.appendChild(c);
        });
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();