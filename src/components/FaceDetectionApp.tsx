"use client";

import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type DragEvent,
  type ChangeEvent,
} from "react";

// =============================================================================
// Types
// =============================================================================

interface FaceRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DetectionResult {
  faces: FaceRect[];
  croppedFaces: string[];
  processingTime: number;
}

type LibraryStatus = "loading" | "ready" | "error";

interface OpenCVParams {
  scaleFactor: number;
  minNeighbors: number;
  minSize: number;
}

interface FaceApiParams {
  inputSize: number;
  scoreThreshold: number;
}

// =============================================================================
// CDN Script Loader
// =============================================================================

const scriptLoadPromises = new Map<string, Promise<void>>();

function loadScript(src: string): Promise<void> {
  const existing = scriptLoadPromises.get(src);
  if (existing) return existing;

  const promise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => {
      scriptLoadPromises.delete(src);
      reject(new Error(`Failed to load: ${src}`));
    };
    document.head.appendChild(script);
  });

  scriptLoadPromises.set(src, promise);
  return promise;
}

// =============================================================================
// OpenCV.js (WASM) - Initialization & Detection
// =============================================================================

const OPENCV_CDN =
  "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.9.0-release.2/dist/opencv.js";
const HAAR_CASCADE_URL =
  "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml";
const HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml";

async function initOpenCV(): Promise<void> {
  await loadScript(OPENCV_CDN);

  let cv = (window as any).cv;
  if (!cv) throw new Error("OpenCV.js failed to load");

  // OpenCV.js 4.5+ uses a factory function pattern:
  // window.cv is a function that returns a Promise resolving to the module.
  if (typeof cv === "function") {
    cv = await cv();
    (window as any).cv = cv;
  } else if (cv instanceof Promise) {
    cv = await cv;
    (window as any).cv = cv;
  } else if (!cv.Mat) {
    // Old-style: wait for WASM runtime initialization
    await new Promise<void>((resolve) => {
      cv["onRuntimeInitialized"] = () => resolve();
    });
  }

  // Load Haar Cascade into the WASM virtual filesystem
  const cvAny = cv as Record<string, (...args: unknown[]) => unknown>;
  let needsLoad = true;
  try {
    const testClassifier = new (cv as any).CascadeClassifier();
    if (testClassifier.load(HAAR_CASCADE_FILE)) {
      needsLoad = false;
    }
    testClassifier.delete();
  } catch {
    // Needs loading
  }

  if (needsLoad) {
    const response = await fetch(HAAR_CASCADE_URL);
    if (!response.ok) throw new Error("Failed to fetch Haar cascade XML");
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    cvAny.FS_createDataFile("/", HAAR_CASCADE_FILE, data, true, false, false);
  }
}

function detectWithOpenCV(imgElement: HTMLImageElement, params: OpenCVParams): DetectionResult {
  const cv = (window as any).cv;

  const startTime = performance.now();

  // Read image at natural size using a temporary canvas to avoid
  // cv.imread using the displayed (CSS-scaled) size
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = imgElement.naturalWidth;
  tmpCanvas.height = imgElement.naturalHeight;
  const tmpCtx = tmpCanvas.getContext("2d")!;
  tmpCtx.drawImage(imgElement, 0, 0, imgElement.naturalWidth, imgElement.naturalHeight);

  const mat = cv.imread(tmpCanvas);
  const gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

  // Equalize histogram for better detection
  cv.equalizeHist(gray, gray);

  // Detect faces using Haar Cascade
  const faces = new cv.RectVector();
  const classifier = new cv.CascadeClassifier();
  classifier.load(HAAR_CASCADE_FILE);
  classifier.detectMultiScale(
    gray,
    faces,
    params.scaleFactor,
    params.minNeighbors,
    0, // flags
    new cv.Size(params.minSize, params.minSize),
    new cv.Size(0, 0) // maxSize (no limit)
  );

  const endTime = performance.now();

  // Extract face rectangles (coordinates are now in natural image space)
  const faceRects: FaceRect[] = [];
  for (let i = 0; i < faces.size(); i++) {
    const face = faces.get(i);
    faceRects.push({
      x: face.x,
      y: face.y,
      width: face.width,
      height: face.height,
    });
  }

  // Crop faces
  const croppedFaces = cropFaces(imgElement, faceRects);

  // Cleanup
  mat.delete();
  gray.delete();
  faces.delete();
  classifier.delete();

  return {
    faces: faceRects,
    croppedFaces,
    processingTime: endTime - startTime,
  };
}

// =============================================================================
// face-api.js (WebGL) - Initialization & Detection
// =============================================================================

const FACE_API_CDN =
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/dist/face-api.js";
const FACE_API_MODEL_URL =
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model";

async function initFaceApi(): Promise<void> {
  await loadScript(FACE_API_CDN);

  const faceapi = (window as any).faceapi;
  if (!faceapi) throw new Error("face-api.js failed to load");

  // Load TinyFaceDetector model (lightweight, fast)
  await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL);
}

async function detectWithFaceApi(
  imgElement: HTMLImageElement,
  params: FaceApiParams
): Promise<DetectionResult> {
  const faceapi = (window as any).faceapi;

  const startTime = performance.now();

  const detections = await faceapi.detectAllFaces(
    imgElement,
    new faceapi.TinyFaceDetectorOptions({
      inputSize: params.inputSize,
      scoreThreshold: params.scoreThreshold,
    })
  );

  const endTime = performance.now();

  const faceRects: FaceRect[] = detections.map(
    (d: { box: { x: number; y: number; width: number; height: number } }) => ({
      x: Math.round(d.box.x),
      y: Math.round(d.box.y),
      width: Math.round(d.box.width),
      height: Math.round(d.box.height),
    })
  );

  const croppedFaces = cropFaces(imgElement, faceRects);

  return {
    faces: faceRects,
    croppedFaces,
    processingTime: endTime - startTime,
  };
}

// =============================================================================
// Utility: Face Cropping
// =============================================================================

function cropFaces(img: HTMLImageElement, faces: FaceRect[]): string[] {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];

  return faces.map((face) => {
    // Add 25% padding around the face
    const padding = Math.max(face.width, face.height) * 0.25;
    const x = Math.max(0, Math.round(face.x - padding));
    const y = Math.max(0, Math.round(face.y - padding));
    const w = Math.min(
      img.naturalWidth - x,
      Math.round(face.width + padding * 2)
    );
    const h = Math.min(
      img.naturalHeight - y,
      Math.round(face.height + padding * 2)
    );

    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, x, y, w, h, 0, 0, w, h);

    return canvas.toDataURL("image/png");
  });
}

// =============================================================================
// Utility: Draw Detection Boxes on Canvas
// =============================================================================

function drawDetections(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  faces: FaceRect[],
  color: string
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Scale canvas to match image natural size
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);

  faces.forEach((face, i) => {
    // Semi-transparent fill
    ctx.fillStyle = color + "20";
    ctx.fillRect(face.x, face.y, face.width, face.height);

    // Border
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.round(img.naturalWidth / 300));
    ctx.strokeRect(face.x, face.y, face.width, face.height);

    // Label
    const fontSize = Math.max(14, Math.round(img.naturalWidth / 40));
    ctx.font = `bold ${fontSize}px sans-serif`;
    const label = `#${i + 1}`;
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = color;
    ctx.fillRect(face.x, face.y - fontSize - 4, textWidth + 8, fontSize + 4);

    ctx.fillStyle = "#fff";
    ctx.fillText(label, face.x + 4, face.y - 4);
  });
}

// =============================================================================
// Main Component
// =============================================================================

export default function FaceDetectionApp() {
  // Library loading states
  const [opencvStatus, setOpencvStatus] = useState<LibraryStatus>("loading");
  const [faceApiStatus, setFaceApiStatus] = useState<LibraryStatus>("loading");

  // Image state
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Detection results
  const [opencvResult, setOpencvResult] = useState<DetectionResult | null>(
    null
  );
  const [faceApiResult, setFaceApiResult] = useState<DetectionResult | null>(
    null
  );

  // UI states
  const [isDetecting, setIsDetecting] = useState(false);
  const [opencvError, setOpencvError] = useState<string | null>(null);
  const [faceApiError, setFaceApiError] = useState<string | null>(null);

  // Detection parameters
  const [opencvParams, setOpencvParams] = useState<OpenCVParams>({
    scaleFactor: 1.1,
    minNeighbors: 9,
    minSize: 80,
  });
  const [faceApiParams, setFaceApiParams] = useState<FaceApiParams>({
    inputSize: 704,
    scoreThreshold: 0.3,
  });
  const [showParams, setShowParams] = useState(false);

  // Refs
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const opencvCanvasRef = useRef<HTMLCanvasElement>(null);
  const faceApiCanvasRef = useRef<HTMLCanvasElement>(null);

  // Load libraries on mount
  useEffect(() => {
    initOpenCV()
      .then(() => setOpencvStatus("ready"))
      .catch((err) => {
        console.error("OpenCV init error:", err);
        setOpencvStatus("error");
      });

    initFaceApi()
      .then(() => setFaceApiStatus("ready"))
      .catch((err) => {
        console.error("face-api.js init error:", err);
        setFaceApiStatus("error");
      });
  }, []);

  // Handle file selection
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setOpencvResult(null);
    setFaceApiResult(null);
    setOpencvError(null);
    setFaceApiError(null);
  }, []);

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  // Run face detection with both methods
  const runDetection = useCallback(async () => {
    if (!imageRef.current) return;
    setIsDetecting(true);
    setOpencvResult(null);
    setFaceApiResult(null);
    setOpencvError(null);
    setFaceApiError(null);

    const img = imageRef.current;

    // Run both detections in parallel
    const [opencvRes, faceApiRes] = await Promise.allSettled([
      (async () => {
        if (opencvStatus !== "ready") throw new Error("OpenCV not loaded");
        return detectWithOpenCV(img, opencvParams);
      })(),
      (async () => {
        if (faceApiStatus !== "ready") throw new Error("face-api.js not loaded");
        return detectWithFaceApi(img, faceApiParams);
      })(),
    ]);

    if (opencvRes.status === "fulfilled") {
      setOpencvResult(opencvRes.value);
    } else {
      setOpencvError(opencvRes.reason?.message || "Detection failed");
    }

    if (faceApiRes.status === "fulfilled") {
      setFaceApiResult(faceApiRes.value);
    } else {
      setFaceApiError(faceApiRes.reason?.message || "Detection failed");
    }

    setIsDetecting(false);
  }, [opencvStatus, faceApiStatus, opencvParams, faceApiParams]);

  // Draw detections on canvas after React renders the canvas elements
  useEffect(() => {
    if (opencvResult && opencvCanvasRef.current && imageRef.current) {
      drawDetections(
        opencvCanvasRef.current,
        imageRef.current,
        opencvResult.faces,
        "#22c55e"
      );
    }
  }, [opencvResult]);

  useEffect(() => {
    if (faceApiResult && faceApiCanvasRef.current && imageRef.current) {
      drawDetections(
        faceApiCanvasRef.current,
        imageRef.current,
        faceApiResult.faces,
        "#3b82f6"
      );
    }
  }, [faceApiResult]);

  // Download cropped face
  const downloadFace = (dataUrl: string, index: number, method: string) => {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `face_${method}_${index + 1}.png`;
    a.click();
  };

  // Status badge
  const StatusBadge = ({
    status,
    label,
  }: {
    status: LibraryStatus;
    label: string;
  }) => (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
        status === "ready"
          ? "bg-green-100 text-green-700"
          : status === "loading"
            ? "bg-yellow-100 text-yellow-700 loading-pulse"
            : "bg-red-100 text-red-700"
      }`}
    >
      <span
        className={`w-2 h-2 rounded-full ${
          status === "ready"
            ? "bg-green-500"
            : status === "loading"
              ? "bg-yellow-500"
              : "bg-red-500"
        }`}
      />
      {label}:{" "}
      {status === "ready"
        ? "Ready"
        : status === "loading"
          ? "Loading..."
          : "Error"}
    </span>
  );

  const maxTime = Math.max(
    opencvResult?.processingTime ?? 0,
    faceApiResult?.processingTime ?? 0
  );

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center mb-6 sm:mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-2">
          Face Detection: WASM vs WebGL
        </h1>
        <p className="text-sm sm:text-base text-gray-600 mb-4 px-2">
          OpenCV.js (WASM / Haar Cascade) vs face-api.js (WebGL /
          TinyFaceDetector)
        </p>
        <div className="flex justify-center gap-2 sm:gap-3 flex-wrap">
          <StatusBadge status={opencvStatus} label="OpenCV WASM" />
          <StatusBadge status={faceApiStatus} label="face-api.js" />
        </div>
      </div>

      {/* Detection Parameters */}
      <div className="mb-6">
        <button
          onClick={() => setShowParams((v) => !v)}
          className="flex items-center gap-2 mx-auto px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <svg
            className={`h-4 w-4 transition-transform ${showParams ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Detection Parameters
        </button>

        {showParams && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* OpenCV Parameters */}
            <div className="bg-white rounded-xl shadow p-4 border border-green-200">
              <h3 className="font-semibold text-green-700 mb-3 text-sm">OpenCV.js (WASM)</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <label>scaleFactor</label>
                    <span className="font-mono">{opencvParams.scaleFactor.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="1.01"
                    max="1.5"
                    step="0.01"
                    value={opencvParams.scaleFactor}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, scaleFactor: parseFloat(e.target.value) }))}
                    className="w-full accent-green-500"
                  />
                  <p className="text-xs text-gray-400">Lower → more accurate / slower. Higher → faster / coarser</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <label>minNeighbors</label>
                    <span className="font-mono">{opencvParams.minNeighbors}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="1"
                    value={opencvParams.minNeighbors}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, minNeighbors: parseInt(e.target.value) }))}
                    className="w-full accent-green-500"
                  />
                  <p className="text-xs text-gray-400">Higher → fewer false positives. Lower → fewer missed faces</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <label>minSize (px)</label>
                    <span className="font-mono">{opencvParams.minSize}</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="300"
                    step="10"
                    value={opencvParams.minSize}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, minSize: parseInt(e.target.value) }))}
                    className="w-full accent-green-500"
                  />
                  <p className="text-xs text-gray-400">Min face size to detect. Higher → ignores small faces</p>
                </div>
              </div>
            </div>

            {/* face-api.js Parameters */}
            <div className="bg-white rounded-xl shadow p-4 border border-blue-200">
              <h3 className="font-semibold text-blue-700 mb-3 text-sm">face-api.js (WebGL)</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <label>inputSize</label>
                    <span className="font-mono">{faceApiParams.inputSize}</span>
                  </div>
                  <input
                    type="range"
                    min="128"
                    max="1024"
                    step="32"
                    value={faceApiParams.inputSize}
                    onChange={(e) => setFaceApiParams((p) => ({ ...p, inputSize: parseInt(e.target.value) }))}
                    className="w-full accent-blue-500"
                  />
                  <p className="text-xs text-gray-400">Higher → more accurate / slower. Lower → faster / more misses</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <label>scoreThreshold</label>
                    <span className="font-mono">{faceApiParams.scoreThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={faceApiParams.scoreThreshold}
                    onChange={(e) => setFaceApiParams((p) => ({ ...p, scoreThreshold: parseFloat(e.target.value) }))}
                    className="w-full accent-blue-500"
                  />
                  <p className="text-xs text-gray-400">Lower → fewer missed faces. Higher → fewer false positives</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Upload Area */}
      <div className="mb-6">
        <div
          className={`drop-zone ${isDragOver ? "drag-over" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {imageUrl ? (
            <div className="relative">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={imageRef}
                src={imageUrl}
                alt="Uploaded"
                className="max-h-80 mx-auto rounded-lg"
                crossOrigin="anonymous"
              />
              <p className="mt-3 text-sm text-gray-500">
                Click or drag to change image
              </p>
            </div>
          ) : (
            <div className="py-12">
              <svg
                className="mx-auto h-12 w-12 sm:h-16 sm:w-16 text-gray-400 mb-3 sm:mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p className="text-base sm:text-lg text-gray-600 font-medium">
                Drop an image here, or click to select
              </p>
              <p className="text-xs sm:text-sm text-gray-400 mt-1">
                JPG, PNG, WebP supported
              </p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>
      </div>

      {/* Detect Button */}
      {imageUrl && (
        <div className="text-center mb-8">
          <button
            onClick={runDetection}
            disabled={
              isDetecting ||
              (opencvStatus !== "ready" && faceApiStatus !== "ready")
            }
            className="w-full sm:w-auto px-8 py-3 bg-gray-900 text-white rounded-lg font-medium text-base sm:text-lg hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isDetecting ? (
              <span className="flex items-center gap-2">
                <svg
                  className="animate-spin h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Detecting...
              </span>
            ) : (
              "Detect Faces"
            )}
          </button>
        </div>
      )}

      {/* Detection Results - Side by Side */}
      {(opencvResult || faceApiResult || opencvError || faceApiError) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* OpenCV WASM Panel */}
          <div className="detection-panel border-2 border-green-200">
            <div className="flex items-center gap-2 mb-3 sm:mb-4">
              <span className="w-3 h-3 rounded-full bg-green-500" />
              <h2 className="text-lg sm:text-xl font-bold text-gray-900">
                OpenCV.js (WASM)
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">
              Haar Cascade Classifier
            </p>

            {opencvError ? (
              <p className="text-red-600">{opencvError}</p>
            ) : opencvResult ? (
              <>
                <div className="flex gap-2 sm:gap-4 mb-3 sm:mb-4 text-xs sm:text-sm">
                  <span className="bg-green-100 text-green-700 px-2 sm:px-3 py-1 rounded-full font-medium">
                    {opencvResult.processingTime.toFixed(1)}ms
                  </span>
                  <span className="bg-gray-100 text-gray-700 px-2 sm:px-3 py-1 rounded-full font-medium">
                    {opencvResult.faces.length} face
                    {opencvResult.faces.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {/* Detection overlay */}
                <div className="mb-4 bg-gray-100 rounded-lg overflow-hidden">
                  <canvas
                    ref={opencvCanvasRef}
                    className="w-full h-auto"
                  />
                </div>

                {/* Cropped faces */}
                {opencvResult.croppedFaces.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">
                      Cropped Faces
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
                      {opencvResult.croppedFaces.map((face, i) => (
                        <div key={i} className="face-card group">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={face}
                            alt={`Face ${i + 1}`}
                            className="w-full h-auto"
                          />
                          <button
                            onClick={() => downloadFace(face, i, "opencv")}
                            className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                          >
                            <svg
                              className="h-6 w-6 text-white"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                              />
                            </svg>
                          </button>
                          <span className="absolute top-1 left-1 bg-green-500 text-white text-xs px-1.5 py-0.5 rounded">
                            #{i + 1}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {opencvResult.faces.length === 0 && (
                  <p className="text-gray-500 text-center py-4">
                    No faces detected
                  </p>
                )}
              </>
            ) : null}
          </div>

          {/* face-api.js WebGL Panel */}
          <div className="detection-panel border-2 border-blue-200">
            <div className="flex items-center gap-2 mb-3 sm:mb-4">
              <span className="w-3 h-3 rounded-full bg-blue-500" />
              <h2 className="text-lg sm:text-xl font-bold text-gray-900">
                face-api.js (WebGL)
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">
              TinyFaceDetector (Neural Network)
            </p>

            {faceApiError ? (
              <p className="text-red-600">{faceApiError}</p>
            ) : faceApiResult ? (
              <>
                <div className="flex gap-2 sm:gap-4 mb-3 sm:mb-4 text-xs sm:text-sm">
                  <span className="bg-blue-100 text-blue-700 px-2 sm:px-3 py-1 rounded-full font-medium">
                    {faceApiResult.processingTime.toFixed(1)}ms
                  </span>
                  <span className="bg-gray-100 text-gray-700 px-2 sm:px-3 py-1 rounded-full font-medium">
                    {faceApiResult.faces.length} face
                    {faceApiResult.faces.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {/* Detection overlay */}
                <div className="mb-4 bg-gray-100 rounded-lg overflow-hidden">
                  <canvas
                    ref={faceApiCanvasRef}
                    className="w-full h-auto"
                  />
                </div>

                {/* Cropped faces */}
                {faceApiResult.croppedFaces.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">
                      Cropped Faces
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
                      {faceApiResult.croppedFaces.map((face, i) => (
                        <div key={i} className="face-card group">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={face}
                            alt={`Face ${i + 1}`}
                            className="w-full h-auto"
                          />
                          <button
                            onClick={() => downloadFace(face, i, "faceapi")}
                            className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                          >
                            <svg
                              className="h-6 w-6 text-white"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                              />
                            </svg>
                          </button>
                          <span className="absolute top-1 left-1 bg-blue-500 text-white text-xs px-1.5 py-0.5 rounded">
                            #{i + 1}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {faceApiResult.faces.length === 0 && (
                  <p className="text-gray-500 text-center py-4">
                    No faces detected
                  </p>
                )}
              </>
            ) : null}
          </div>
        </div>
      )}

      {/* Speed Comparison */}
      {opencvResult && faceApiResult && (
        <div className="bg-white rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8">
          <h2 className="text-lg sm:text-xl font-bold text-gray-900 mb-3 sm:mb-4">
            Speed Comparison
          </h2>

          <div className="space-y-4">
            {/* OpenCV bar */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">
                  OpenCV.js (WASM)
                </span>
                <span className="text-sm font-bold text-green-700">
                  {opencvResult.processingTime.toFixed(1)}ms
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-lg overflow-hidden">
                <div
                  className="speed-bar bg-green-500"
                  style={{
                    width:
                      maxTime > 0
                        ? `${Math.max(5, (opencvResult.processingTime / maxTime) * 100)}%`
                        : "5%",
                  }}
                >
                  {opencvResult.faces.length} face
                  {opencvResult.faces.length !== 1 ? "s" : ""}
                </div>
              </div>
            </div>

            {/* face-api.js bar */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">
                  face-api.js (WebGL)
                </span>
                <span className="text-sm font-bold text-blue-700">
                  {faceApiResult.processingTime.toFixed(1)}ms
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-lg overflow-hidden">
                <div
                  className="speed-bar bg-blue-500"
                  style={{
                    width:
                      maxTime > 0
                        ? `${Math.max(5, (faceApiResult.processingTime / maxTime) * 100)}%`
                        : "5%",
                  }}
                >
                  {faceApiResult.faces.length} face
                  {faceApiResult.faces.length !== 1 ? "s" : ""}
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">
              {opencvResult.processingTime < faceApiResult.processingTime ? (
                <>
                  <span className="font-bold text-green-700">
                    OpenCV WASM
                  </span>{" "}
                  was{" "}
                  <span className="font-bold">
                    {(
                      faceApiResult.processingTime /
                      opencvResult.processingTime
                    ).toFixed(1)}
                    x
                  </span>{" "}
                  faster than face-api.js WebGL
                </>
              ) : opencvResult.processingTime >
                faceApiResult.processingTime ? (
                <>
                  <span className="font-bold text-blue-700">
                    face-api.js WebGL
                  </span>{" "}
                  was{" "}
                  <span className="font-bold">
                    {(
                      opencvResult.processingTime /
                      faceApiResult.processingTime
                    ).toFixed(1)}
                    x
                  </span>{" "}
                  faster than OpenCV WASM
                </>
              ) : (
                <>Both methods had the same processing time</>
              )}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Note: First run may be slower due to JIT compilation warmup.
              Re-run for more accurate comparison.
            </p>
          </div>
        </div>
      )}

      {/* Technical Details */}
      <div className="bg-white rounded-xl shadow-lg p-4 sm:p-6">
        <h2 className="text-lg sm:text-xl font-bold text-gray-900 mb-3 sm:mb-4">
          Technical Details
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 text-xs sm:text-sm">
          <div>
            <h3 className="font-semibold text-green-700 mb-2">
              OpenCV.js (WASM)
            </h3>
            <ul className="space-y-1 text-gray-600">
              <li>
                <span className="font-medium">Engine:</span> OpenCV 4.9.0
                compiled to WebAssembly
              </li>
              <li>
                <span className="font-medium">Algorithm:</span> Haar Cascade
                Classifier
              </li>
              <li>
                <span className="font-medium">Approach:</span> Classical
                computer vision (feature-based)
              </li>
              <li>
                <span className="font-medium">Execution:</span> CPU via WASM
              </li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-blue-700 mb-2">
              face-api.js (WebGL)
            </h3>
            <ul className="space-y-1 text-gray-600">
              <li>
                <span className="font-medium">Engine:</span> TensorFlow.js
                with WebGL backend
              </li>
              <li>
                <span className="font-medium">Algorithm:</span>{" "}
                TinyFaceDetector (CNN-based)
              </li>
              <li>
                <span className="font-medium">Approach:</span> Deep learning
                (neural network)
              </li>
              <li>
                <span className="font-medium">Execution:</span> GPU via WebGL
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
