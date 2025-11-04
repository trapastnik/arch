export const MODEL_URL = 'web_model/model.onnx';
export const MAP_URL   = 'class_mapping.json';

export const INPUT_SIZE = 224;        // размер входа модели
export const LOOP_INTERVAL_MS = 150;  // частота инференса
export const EDGE_SIZE = 160;         // масштаб для быстрого поиска рамки

// Режимы кадрирования
export const CROP_MODE = {
  FULL: 'full',       // весь кадр resize
  CENTER: 'center',   // квадрат из центра
  AUTO: 'auto'        // детектор рамки (умная рамка)
};

// Режимы нормализации
export const NORM_MODE = {
  NEG1TO1: 'neg1to1',     // x/127.5 - 1
  ZERO1: 'zero1',         // x/255
  IMAGENET: 'imagenet'    // (x - mean)/std
};

// ImageNet mean/std для RGB
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

// Дефолтные настройки
export const DEFAULT_CROP_MODE = CROP_MODE.AUTO;
export const DEFAULT_NORM_MODE = NORM_MODE.NEG1TO1;
export const DEFAULT_ROI_PAD = 0.08;    // паддинг вокруг ROI (8%)
export const DEFAULT_BOX_EMA = 0.35;    // сглаживание рамки (0 = без сглаживания)
