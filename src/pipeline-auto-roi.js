/* src/pipeline-auto-roi.js
 * Авто-ROI (контуры -> четыре угла) -> перспективное выравнивание -> anti-moire -> resize+normalize -> ONNX
 * + EMA сглаживание и temperature scaling. Ничего из существующего кода не ломаем.
 */

(async function () {
  // === Настройки под твой проект ===
  const MODEL_URL = 'web_model/model.onnx';
  const INPUT_SIZE = 224;
  const MEAN = [0.485, 0.456, 0.406];   // если обучалось на ImageNet
  const STD  = [0.229, 0.224, 0.225];
  const TEMP = 1.5;                     // temperature scaling
  const EMA_ALPHA = 0.7;                // сглаживание вероятностей
  const CONF_THRESH = 0.75;             // порог показа (после EMA)

  // Важно: Katarsis лежат в выходах модели по индексам 3,4,5
  const KATARSIS_INDEXES = [3,4,5];

  // Ждём готовности OpenCV
  await new Promise(res => {
    if (window.cv && cv.getBuildInformation) return res();
    const timer = setInterval(() => { if (window.cv && cv.getBuildInformation) { clearInterval(timer); res(); } }, 50);
  });

  // Загружаем class_mapping.json (любой формат: массив или объект)
  const mappingRaw = await fetch('class_mapping.json').then(r=>r.json()).catch(()=>null);
  const id2label = mappingRaw
    ? (Array.isArray(mappingRaw) ? mappingRaw
       : Object.keys(mappingRaw).sort((a,b)=>(+a)-(+b)).map(k=>mappingRaw[k]))
    : [];

  // Создаём сессию ONNX
  const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm','webgl'] // webgpu можно добавить если нужно
  });

  // === Небольшие утилиты ===
  function softmaxWithT(logits, T=1.0) {
    const scaled = logits.map(v => v / T);
    const m = Math.max(...scaled);
    const exps = scaled.map(v => Math.exp(v - m));
    const sum = exps.reduce((a,b)=>a+b, 0);
    return exps.map(v => v / sum);
  }

  function emaInit(n) { return { v: new Float32Array(n), ready:false }; }
  function emaUpdate(state, probs, alpha=0.7) {
    if (!state.ready) { state.v.set(probs); state.ready = true; }
    else for (let i=0;i<probs.length;i++) state.v[i] = alpha*state.v[i] + (1-alpha)*probs[i];
    return state.v;
  }

  function orderQuad(pts) {
    const s = pts.map(p=>p.x+p.y), d = pts.map(p=>p.x-p.y);
    const tl = pts[s.indexOf(Math.min(...s))];
    const br = pts[s.indexOf(Math.max(...s))];
    const tr = pts[d.indexOf(Math.min(...d))];
    const bl = pts[d.indexOf(Math.max(...d))];
    return [tl,tr,br,bl];
  }

  // Поиск квадратов (контуры -> аппроксимация)
  function detectQuads(mat, opts={}) {
    const { canny1=70, canny2=180, eps=0.02, minAreaRatio=0.03, maxSideRatio=1.7 } = opts;
    const w = mat.cols, h = mat.rows, minArea = minAreaRatio * w * h;

    let gray = new cv.Mat(); cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    cv.medianBlur(gray, gray, 3);
    cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0);

    let edges = new cv.Mat();
    cv.Canny(gray, edges, canny1, canny2);

    let contours = new cv.MatVector(), hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const quads = [];
    for (let i=0; i<contours.size(); i++) {
      const cnt = contours.get(i);
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, eps*peri, true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const area = cv.contourArea(approx);
        if (area >= minArea) {
          const pts = [];
          for (let j=0;j<4;j++) {
            const p = approx.intPtr(j);
            pts.push({x:p[0], y:p[1]});
          }
          const sides = [
            Math.hypot(pts[1].x-pts[0].x, pts[1].y-pts[0].y),
            Math.hypot(pts[2].x-pts[1].x, pts[2].y-pts[1].y),
            Math.hypot(pts[3].x-pts[2].x, pts[3].y-pts[2].y),
            Math.hypot(pts[0].x-pts[3].x, pts[0].y-pts[3].y),
          ];
          const ratio = Math.max(...sides) / (Math.min(...sides)+1e-6);
          if (ratio < maxSideRatio) quads.push(pts);
        }
      }
      approx.delete(); cnt.delete();
    }
    gray.delete(); edges.delete(); contours.delete(); hierarchy.delete();

    // сортируем по площади по убыванию
    quads.sort((a,b)=>{
      const area = (qq)=> {
        const m = cv.matFromArray(qq.length,1,cv.CV_32SC2, qq.flatMap(p=>[p.x,p.y]));
        const A = cv.contourArea(m); m.delete(); return A;
      };
      return area(b) - area(a);
    });
    return quads;
  }

  // Перспективная коррекция квадрата -> canvas ROI
  function warpQuadToCanvas(srcCanvas, quad, outSize=256) {
    const [tl,tr,br,bl] = orderQuad(quad);
    const src = cv.imread(srcCanvas);
    const dst = new cv.Mat();

    const srcTri = cv.matFromArray(4,1,cv.CV_32FC2,[tl.x,tl.y, tr.x,tr.y, br.x,br.y, bl.x,bl.y]);
    const dstTri = cv.matFromArray(4,1,cv.CV_32FC2,[0,0, outSize,0, outSize,outSize, 0,outSize]);

    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    cv.warpPerspective(src, dst, M, new cv.Size(outSize,outSize), cv.INTER_CUBIC, cv.BORDER_REPLICATE);

    const out = document.createElement('canvas');
    out.width = outSize; out.height = outSize;
    cv.imshow(out, dst);

    src.delete(); dst.delete(); srcTri.delete(); dstTri.delete(); M.delete();
    return out;
  }

  // Лёгкий anti-moire перед ресайзом (2D box/gauss через Canvas API можно упростить)
  function applyAntiMoiré(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently:true });
    // маленький трюк: downscale -> upscale сглаживает муар
    const tmp = document.createElement('canvas');
    tmp.width = Math.round(canvas.width * 0.75);
    tmp.height = Math.round(canvas.height * 0.75);
    tmp.getContext('2d').drawImage(canvas, 0,0, tmp.width, tmp.height);
    ctx.clearRect(0,0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0,0, canvas.width, canvas.height);
    return canvas;
  }

  // Преобразование ROI canvas -> Tensor CHW float32
  function toTensorCHW(canvas, size=INPUT_SIZE) {
    const tmp = document.createElement('canvas');
    tmp.width = size; tmp.height = size;
    tmp.getContext('2d').drawImage(canvas, 0, 0, size, size);

    const { data } = tmp.getContext('2d').getImageData(0,0,size,size);
    const chw = new Float32Array(3*size*size);
    for (let i=0,p=0;i<data.length;i+=4,p++){
      const r = data[i]   / 255;
      const g = data[i+1] / 255;
      const b = data[i+2] / 255;
      chw[p]               = (r - MEAN[0]) / STD[0];
      chw[p + size*size]   = (g - MEAN[1]) / STD[1];
      chw[p + 2*size*size] = (b - MEAN[2]) / STD[2];
    }
    return new ort.Tensor('float32', chw, [1,3,size,size]);
  }

  // ===== Основной рантайм: берём текущее видео-канвас, ищем и классифицируем =====
  // Предположим, что у тебя уже есть <video id="camera"> и <canvas id="frame"> или подобные элементы.
  // Если нет — можно быстро добавить их в index.html. Здесь я возьму самое простое имя:
  const video = document.querySelector('video') || document.getElementById('camera');
  if (!video) {
    console.warn('[Auto-ROI] Не найден <video>. Пайплайн ждать не будет.');
    return;
  }

  const overlay = document.getElementById('overlay') || (()=> {
    const c = document.createElement('canvas'); c.id='overlay';
    c.style.position='absolute'; c.style.left='0'; c.style.top='0'; c.style.pointerEvents='none';
    video.parentElement?.appendChild(c); return c;
  })();

  const ctxOv = overlay.getContext('2d');

  const emaByTrack = new Map(); // key -> EMA state

  async function tick() {
    if (video.readyState >= 2) {
      overlay.width = video.videoWidth; overlay.height = video.videoHeight;

      // Рисуем кадр в мат для OpenCV
      const frame = document.createElement('canvas');
      frame.width = video.videoWidth; frame.height = video.videoHeight;
      frame.getContext('2d').drawImage(video, 0, 0, frame.width, frame.height);

      const src = cv.imread(frame);
      const quads = detectQuads(src, { canny1:70, canny2:180, eps:0.02, minAreaRatio:0.03, maxSideRatio:1.7 });
      src.delete();

      // Оверлей
      ctxOv.clearRect(0,0,overlay.width,overlay.height);
      ctxOv.lineWidth = 3; ctxOv.strokeStyle = 'lime';

      for (let qi=0; qi<Math.min(quads.length, 4); qi++) {
        const q = quads[qi];
        // Рисуем контур
        ctxOv.beginPath();
        ctxOv.moveTo(q[0].x,q[0].y);
        for (let j=1;j<4;j++) ctxOv.lineTo(q[j].x,q[j].y);
        ctxOv.closePath(); ctxOv.stroke();

        // Перспективная коррекция -> анти-муар -> тензор
        let roiCanvas = warpQuadToCanvas(frame, q, 256);
        roiCanvas = applyAntiMoiré(roiCanvas);
        const tensor = toTensorCHW(roiCanvas, INPUT_SIZE);

        // Инференс
        const out = await session.run({ [session.inputNames[0]]: tensor });
        const logits = Array.from(out[session.outputNames[0]].data);
        const probs = softmaxWithT(logits, TEMP);

        // EMA по этому ROI (простая привязка по индексу qi)
        const key = `roi_${qi}`;
        if (!emaByTrack.has(key)) emaByTrack.set(key, emaInit(probs.length));
        const smooth = emaUpdate(emaByTrack.get(key), probs, EMA_ALPHA);

        // Топ-к, показываем только если уверенность достаточна
        const top = Array.from(smooth).map((p,i)=>({i,p})).sort((a,b)=>b.p-a.p)[0];
        const label = id2label[top.i] ?? `class_${top.i}`;
        const confident = top.p >= CONF_THRESH;

        // Если хочешь показывать только Katarsis, раскомментируй:
        // if (!KATARSIS_INDEXES.includes(top.i)) continue;

        // Надпись
        ctxOv.fillStyle = confident ? 'rgba(0,128,0,0.7)' : 'rgba(0,0,0,0.5)';
        ctxOv.font = '16px sans-serif';
        ctxOv.fillText(`${label} ${(top.p*100).toFixed(1)}%`, q[0].x+6, q[0].y+18);
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();