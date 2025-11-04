import { $, logDiag } from './ui.js';

let stream = null;

export async function startCamera(usingBack){
  if (stream) await stopCamera();
  const constraints = { video: { facingMode: usingBack ? { ideal: 'environment' } : 'user' }, audio: false };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  const video = $('#video');
  video.setAttribute('playsinline',''); // iOS
  video.srcObject = stream;
  video.muted = true;
  await video.play();
  return video;
}

export async function stopCamera(){
  try{
    if (stream){ for (const t of stream.getTracks()) t.stop(); }
  } catch(e){ logDiag(e?.message || e); }
  stream = null;
  const video = $('#video');
  if (video) video.srcObject = null;
}

export function hasStream(){ return !!stream; }
