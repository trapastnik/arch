import { MODEL_URL, MAP_URL, INPUT_SIZE } from './constants.js';
import { logDiag, setStatus } from './ui.js';

// загрузка модели с прогрессом (в Uint8Array)
async function fetchModelWithProgress(url, onProgress){
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} while fetching model`);
  const total = +resp.headers.get('Content-Length') || 0;
  const reader = resp.body?.getReader?.();
  if (!reader) {
    const ab = await resp.arrayBuffer();
    onProgress?.(1);
    return new Uint8Array(ab);
  }
  const chunks = [];
  let rec = 0;
  for(;;){
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    rec += value.byteLength;
    if (total) onProgress?.(rec/total);
  }
  const blob = new Blob(chunks);
  const ab = await blob.arrayBuffer();
  onProgress?.(1);
  return new Uint8Array(ab);
}

export async function loadModel(){
  setStatus('Загрузка рантайма…'); await new Promise(r=>setTimeout(r,40));

  setStatus('Загрузка модели…', 0);
  const modelBytes = await fetchModelWithProgress(MODEL_URL, p => setStatus(`Загрузка модели… ${Math.round(p*100)}%`, p));

  setStatus('Инициализация модели…');
  
  // ⚡ Пытаемся использовать WebGL для ускорения (fallback на wasm)
  // WebGL обычно в 2-5x быстрее на GPU, но не везде стабилен
  let executionProviders = ['webgl', 'wasm'];
  
  // Включаем SIMD и многопоточность если браузер поддерживает
  const sessionOptions = {
    executionProviders,
    graphOptimizationLevel: 'all',
    executionMode: 'parallel',  // параллельное выполнение операций
  };
  
  // Проверяем поддержку WebAssembly SIMD и Threads
  try {
    if (typeof WebAssembly.validate === 'function') {
      // SIMD проверка
      const simdSupported = WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
      ]));
      
      // Threads проверка
      const threadsSupported = typeof SharedArrayBuffer !== 'undefined';
      
      logDiag({ simdSupported, threadsSupported, providers: executionProviders });
      
      if (simdSupported) {
        sessionOptions.extra = { 
          ...sessionOptions.extra,
          wasmPaths: {
            'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm'
          }
        };
      }
    }
  } catch (e) {
    logDiag('SIMD/Threads check failed: ' + (e?.message || e));
  }
  
  const session = await ort.InferenceSession.create(modelBytes, sessionOptions);

  const INPUT_NAME  = session.inputNames?.[0] ?? null;
  const OUTPUT_NAME = session.outputNames?.[0] ?? null;

  // безопасное чтение метаданных
  let dims = null;
  try {
    const meta = session.inputMetadata;
    if (meta){
      if (typeof meta.get === 'function' && INPUT_NAME){
        dims = meta.get(INPUT_NAME)?.dimensions ?? null;
      } else if (INPUT_NAME && meta[INPUT_NAME]){
        dims = meta[INPUT_NAME].dimensions ?? null;
      }
    }
  } catch(e){ logDiag('meta read error: ' + (e?.message || e)); }

  const useNCHW = (Array.isArray(dims) && dims.length === 4 && dims[1] === 3);
  
  // Логируем какой провайдер реально используется
  let actualProvider = 'unknown';
  try {
    // ONNXRuntime Web не всегда предоставляет эту информацию
    actualProvider = executionProviders[0]; // первый доступный
  } catch (e) {
    logDiag('Provider check: ' + (e?.message || e));
  }
  
  logDiag({ 
    inputName: INPUT_NAME, 
    outputName: OUTPUT_NAME, 
    inputDims: dims, 
    useNCHW,
    provider: actualProvider 
  });

  setStatus('Загрузка карты классов…');
  const mapping = await fetch(MAP_URL).then(r=>r.json());

  setStatus('Прогрев модели…');
  try{
    const zeros = new Float32Array(INPUT_SIZE*INPUT_SIZE*3);
    const warm  = useNCHW
      ? new ort.Tensor('float32', zeros, [1,3,INPUT_SIZE,INPUT_SIZE])
      : new ort.Tensor('float32', zeros, [1,INPUT_SIZE,INPUT_SIZE,3]);
    await session.run({ [INPUT_NAME]: warm });
  }catch(e){ logDiag('warmup: '+(e?.message||e)); }

  setStatus('Модель готова. Нажми «Старт камеры».', null);

  return { session, INPUT_NAME, OUTPUT_NAME, useNCHW, mapping };
}
