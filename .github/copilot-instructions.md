## Quick guidance for AI coding agents

This is a small single-page web app for image recognition using ONNX Runtime Web. The goal is to keep edits minimal, preserve UI controls, and avoid changing the inference contract (input size / tensor layout).

- Entry point: `index.html` (loads `onnxruntime-web` from CDN and `src/main.js`).
- Orchestration: `src/main.js` — loads model, starts camera, runs the infer loop, updates UI.
- Model I/O and runtime: `src/model.js` — downloads model bytes, creates `ort.InferenceSession` with executionProviders (`webgl` then `wasm`), does warmup, and fetches `class_mapping.json`.
- Preprocessing: `src/preprocess.js` — implements three crop modes (`full`, `center`, `auto`), three normalization modes (`neg1to1`, `zero1`, `imagenet`), and returns an `ort.Tensor` shaped to either NCHW or NHWC depending on model metadata.
- ROI/Detector: `src/box.js` — fast edge-energy square search that returns a box; EMA smoothing applied if configured.
- Camera helpers: `src/camera.js` — `startCamera` / `stopCamera` wrapping `getUserMedia`.
- UI helpers: `src/ui.js` — lightweight DOM helpers and diagnostics output (element id: diag).

Important files to reference in edits
- `src/constants.js` — input size, LOOP_INTERVAL_MS, EDGE_SIZE, default crop/norm modes and tunables. Adjust here for global behavior.
- `class_mapping.json` — mapping from integer class index to display name; used in UI (`state.mapping[String(idx)]`).
- `web_model/model.onnx` — ONNX model file used by `MODEL_URL`.

Developer workflows / run & debug
- Local dev: serve the repo over HTTP/HTTPS — do NOT open via `file://`. A minimal command used in README: `python3 -m http.server 8000` and open `http://localhost:8000`.
- Production: app is static and intended for GitHub Pages (HTTPS required for camera access on many platforms).
- Debugging: open browser DevTools and/or the in-page Details > Диагностика (element id: diag) to inspect logs produced by the logDiag helper. `model.js` logs model input/output names, dims and provider info.
- Mobile caveats: iOS requires a user gesture to start the camera. The UI already requires clicking the Start button.

Project-specific conventions and gotchas
- Don’t assume tensor layout: `model.js` inspects input metadata and sets `useNCHW`. Always use `preprocessToTensor(..., useNCHW)` which respects that flag.
- The app expects square inputs: `INPUT_SIZE` (224) — any model changes must match this or require coordinated changes in `preprocess.js` and `constants.js`.
- `box.js` searches only square boxes (aspect ratio = 1.0). If you change detector logic, update UI assumptions and ROI padding handling.
- Progress & diagnostics use `setStatus(...)` and `logDiag(...)`. Preserve these calls when adding async flows so the user sees progress.

Integration points / external dependencies
- ONNX Runtime Web is loaded from CDN in `index.html` (`ort.min.js`). Changes to runtime versions should be tested across browsers (WebGL vs WASM differences).
- Model and mapping are fetched via `MODEL_URL` / `MAP_URL` from `constants.js` — these are relative paths under the repo and must be hosted together.

Suggested edits patterns (examples)
- To change inference rate: edit `LOOP_INTERVAL_MS` in `src/constants.js` (e.g., 150 → 250 for slower phones).
- To add a new normalization option: add case in `preprocess.js`, add option to `<select id="selNorm">` in `index.html`, and ensure `main.js` reacts to the new value.
- To experiment with different providers: `src/model.js` sets `executionProviders = ['webgl','wasm']`. Test fallbacks and log provider selection.

Contract & quick checklist for changes that touch inference
- Input: square image of size `INPUT_SIZE` (default 224). Preprocess returns an `ort.Tensor` with dtype `float32` and shape `[1,3,H,W]` (NCHW) or `[1,H,W,3]` (NHWC) depending on `useNCHW`.
- Output: `session.run(...)` expects `{ [INPUT_NAME]: tensor }` and returns logits in `OUTPUT_NAME` used by `preprocess.topkSoftmax`.
- When changing shapes/norm/scaling, update both `constants.js` and `preprocess.js`, and update `README.md`/diagnostics to help runtime debugging.

What to avoid
- Don’t remove progress/diagnostic logs — they are primary debugging hooks for this client-only app.
- Don’t hardcode `useNCHW` layout — rely on session metadata as implemented in `src/model.js`.

If unsure, inspect these files first: `src/main.js`, `src/model.js`, `src/preprocess.js`, `src/box.js`, `src/constants.js`, `index.html`.

Next steps for reviewer
- Tell me if you want stricter linting, type hints (TypeScript), or a tiny test harness for the preprocessing functions — I can add them next.
