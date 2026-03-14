/**
 * Shared image preprocessing for all framework adapters.
 * Loads a user-provided image, resizes to 224x224, and returns
 * normalized pixel data in multiple formats.
 *
 * Preprocessing per model:
 * - TF.js (TFHub):     NHWC [0, 1]
 * - ONNX (HF/Google):  NCHW [-1, 1] (mean=0.5, std=0.5)
 * - LiteRT (TFLite):   NHWC [-1, 1]
 * - Transformers.js:    handles its own preprocessing from dataUrl
 */

const SIZE = 224;

export interface PreprocessedImage {
  /** NHWC layout [1, 224, 224, 3], values in [0, 1] — used by TF.js */
  nhwcZeroOne: Float32Array;
  /** NHWC layout [1, 224, 224, 3], values in [-1, 1] — used by LiteRT */
  nhwcNegOneOne: Float32Array;
  /** NCHW layout [1, 3, 224, 224], values in [-1, 1] — used by ONNX */
  nchwNegOneOne: Float32Array;
  /** Data URL of the resized image — used by Transformers.js */
  dataUrl: string;
}

/**
 * Load an image from a File or URL, resize to 224x224, return preprocessed data.
 */
export async function preprocessImage(source: File | string): Promise<PreprocessedImage> {
  const img = await loadImage(source);
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  const { data } = imageData;

  const nhwcZeroOne = new Float32Array(1 * SIZE * SIZE * 3);
  const nhwcNegOneOne = new Float32Array(1 * SIZE * SIZE * 3);
  const nchwNegOneOne = new Float32Array(1 * 3 * SIZE * SIZE);

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const srcIdx = (y * SIZE + x) * 4; // RGBA
      const pixIdx = y * SIZE + x;

      // [0, 1] range
      const r = data[srcIdx]! / 255;
      const g = data[srcIdx + 1]! / 255;
      const b = data[srcIdx + 2]! / 255;

      // NHWC [0, 1] — for TF.js
      nhwcZeroOne[pixIdx * 3] = r;
      nhwcZeroOne[pixIdx * 3 + 1] = g;
      nhwcZeroOne[pixIdx * 3 + 2] = b;

      // [-1, 1] range: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
      const rNorm = r * 2 - 1;
      const gNorm = g * 2 - 1;
      const bNorm = b * 2 - 1;

      // NHWC [-1, 1] — for LiteRT
      nhwcNegOneOne[pixIdx * 3] = rNorm;
      nhwcNegOneOne[pixIdx * 3 + 1] = gNorm;
      nhwcNegOneOne[pixIdx * 3 + 2] = bNorm;

      // NCHW [-1, 1] — for ONNX
      nchwNegOneOne[0 * SIZE * SIZE + pixIdx] = rNorm;
      nchwNegOneOne[1 * SIZE * SIZE + pixIdx] = gNorm;
      nchwNegOneOne[2 * SIZE * SIZE + pixIdx] = bNorm;
    }
  }

  const dataUrl = canvas.toDataURL('image/png');

  return { nhwcZeroOne, nhwcNegOneOne, nchwNegOneOne, dataUrl };
}

function loadImage(source: File | string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    if (typeof source === 'string') {
      img.src = source;
    } else {
      img.src = URL.createObjectURL(source);
    }
  });
}

/**
 * Generate random dummy preprocessed image (used when no image is uploaded).
 */
export function generateDummyImage(): PreprocessedImage {
  const nhwcZeroOne = new Float32Array(1 * SIZE * SIZE * 3);
  const nhwcNegOneOne = new Float32Array(1 * SIZE * SIZE * 3);
  const nchwNegOneOne = new Float32Array(1 * 3 * SIZE * SIZE);

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const pixIdx = y * SIZE + x;
      const r = Math.random();
      const g = Math.random();
      const b = Math.random();

      nhwcZeroOne[pixIdx * 3] = r;
      nhwcZeroOne[pixIdx * 3 + 1] = g;
      nhwcZeroOne[pixIdx * 3 + 2] = b;

      const rNorm = r * 2 - 1;
      const gNorm = g * 2 - 1;
      const bNorm = b * 2 - 1;

      nhwcNegOneOne[pixIdx * 3] = rNorm;
      nhwcNegOneOne[pixIdx * 3 + 1] = gNorm;
      nhwcNegOneOne[pixIdx * 3 + 2] = bNorm;

      nchwNegOneOne[0 * SIZE * SIZE + pixIdx] = rNorm;
      nchwNegOneOne[1 * SIZE * SIZE + pixIdx] = gNorm;
      nchwNegOneOne[2 * SIZE * SIZE + pixIdx] = bNorm;
    }
  }

  // Render the random noise as a visible image
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const r = nhwcZeroOne[i * 3]! * 255;
    const g = nhwcZeroOne[i * 3 + 1]! * 255;
    const b = nhwcZeroOne[i * 3 + 2]! * 255;
    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  return { nhwcZeroOne, nhwcNegOneOne, nchwNegOneOne, dataUrl: canvas.toDataURL('image/png') };
}
