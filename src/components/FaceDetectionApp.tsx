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
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md border transition-all duration-300 ${
        status === "ready"
          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
          : status === "loading"
            ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
            : "bg-red-500/10 text-red-400 border-red-500/30"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          status === "ready"
            ? "bg-emerald-400"
            : status === "loading"
              ? "bg-amber-400 status-dot-loading"
              : "bg-red-400"
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
      <div className="text-center mb-8 sm:mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-800/60 border border-zinc-700/50 text-zinc-400 text-xs font-medium mb-4 backdrop-blur-sm">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Browser-native inference
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold mb-3 tracking-tight">
          <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Face Detection
          </span>
        </h1>
        <p className="text-sm sm:text-base text-zinc-500 mb-6 max-w-lg mx-auto">
          Haar Cascade <span className="text-zinc-600">·</span> WASM{" "}
          <span className="text-emerald-500/60 mx-1">vs</span>{" "}
          TinyFaceDetector <span className="text-zinc-600">·</span> WebGL
        </p>
        <div className="flex justify-center gap-2 sm:gap-3 flex-wrap">
          <StatusBadge status={opencvStatus} label="OpenCV" />
          <StatusBadge status={faceApiStatus} label="face-api" />
        </div>
      </div>

      {/* Detection Parameters */}
      <div className="mb-6">
        <button
          onClick={() => setShowParams((v) => !v)}
          className="flex items-center gap-2 mx-auto px-4 py-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors rounded-lg hover:bg-zinc-800/50"
        >
          <svg
            className={`h-3.5 w-3.5 transition-transform duration-300 ${showParams ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Detection Parameters
        </button>

        {showParams && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
            {/* OpenCV Parameters */}
            <div className="glass-card panel-green">
              <h3 className="font-semibold text-emerald-400 mb-4 text-xs uppercase tracking-wider">OpenCV.js — WASM</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <label className="text-zinc-400">scaleFactor</label>
                    <span className="mono-value text-emerald-400">{opencvParams.scaleFactor.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="1.01"
                    max="1.5"
                    step="0.01"
                    value={opencvParams.scaleFactor}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, scaleFactor: parseFloat(e.target.value) }))}
                    className="w-full accent-green"
                  />
                  <p className="text-[10px] text-zinc-600 mt-1">Lower → more accurate / slower</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <label className="text-zinc-400">minNeighbors</label>
                    <span className="mono-value text-emerald-400">{opencvParams.minNeighbors}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="1"
                    value={opencvParams.minNeighbors}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, minNeighbors: parseInt(e.target.value) }))}
                    className="w-full accent-green"
                  />
                  <p className="text-[10px] text-zinc-600 mt-1">Higher → fewer false positives</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <label className="text-zinc-400">minSize (px)</label>
                    <span className="mono-value text-emerald-400">{opencvParams.minSize}</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="300"
                    step="10"
                    value={opencvParams.minSize}
                    onChange={(e) => setOpencvParams((p) => ({ ...p, minSize: parseInt(e.target.value) }))}
                    className="w-full accent-green"
                  />
                  <p className="text-[10px] text-zinc-600 mt-1">Min face size to detect</p>
                </div>
              </div>
            </div>

            {/* face-api.js Parameters */}
            <div className="glass-card panel-blue">
              <h3 className="font-semibold text-blue-400 mb-4 text-xs uppercase tracking-wider">face-api.js — WebGL</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <label className="text-zinc-400">inputSize</label>
                    <span className="mono-value text-blue-400">{faceApiParams.inputSize}</span>
                  </div>
                  <input
                    type="range"
                    min="128"
                    max="1024"
                    step="32"
                    value={faceApiParams.inputSize}
                    onChange={(e) => setFaceApiParams((p) => ({ ...p, inputSize: parseInt(e.target.value) }))}
                    className="w-full accent-blue"
                  />
                  <p className="text-[10px] text-zinc-600 mt-1">Higher → more accurate / slower</p>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <label className="text-zinc-400">scoreThreshold</label>
                    <span className="mono-value text-blue-400">{faceApiParams.scoreThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={faceApiParams.scoreThreshold}
                    onChange={(e) => setFaceApiParams((p) => ({ ...p, scoreThreshold: parseFloat(e.target.value) }))}
                    className="w-full accent-blue"
                  />
                  <p className="text-[10px] text-zinc-600 mt-1">Lower → fewer missed faces</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Upload Area */}
      <div className="mb-6">
        <div
          className={`drop-zone ${isDragOver ? "drag-over" : ""} ${isDetecting ? "scan-effect" : ""}`}
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
              <p className="mt-3 text-xs text-zinc-600">
                Click or drag to change image
              </p>
            </div>
          ) : (
            <div className="py-12 sm:py-16">
              <div className="mx-auto w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center mb-4 sm:mb-5">
                <svg
                  className="h-8 w-8 sm:h-10 sm:w-10 text-zinc-500"
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
              </div>
              <p className="text-sm sm:text-base text-zinc-400 font-medium">
                Drop an image here, or click to select
              </p>
              <p className="text-xs text-zinc-600 mt-1.5">
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
        <div className="text-center mb-8 animate-fade-in">
          <button
            onClick={runDetection}
            disabled={
              isDetecting ||
              (opencvStatus !== "ready" && faceApiStatus !== "ready")
            }
            className="btn-detect w-full sm:w-auto px-10 py-3.5 text-white rounded-xl font-semibold text-sm sm:text-base tracking-wide"
          >
            {isDetecting ? (
              <span className="flex items-center justify-center gap-2.5">
                <svg
                  className="animate-spin h-4 w-4"
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
                Processing...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Detect Faces
              </span>
            )}
          </button>
        </div>
      )}

      {/* Detection Results - Side by Side */}
      {(opencvResult || faceApiResult || opencvError || faceApiError) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-8 animate-fade-in">
          {/* OpenCV WASM Panel */}
          <div className={`glass-card panel-green ${opencvResult ? "has-result" : ""}`}>
            <div className="flex items-center gap-2.5 mb-3">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
              <h2 className="text-base sm:text-lg font-bold text-zinc-200">
                OpenCV.js
              </h2>
              <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-medium ml-auto">WASM</span>
            </div>
            <p className="text-xs text-zinc-500 mb-3">
              Haar Cascade Classifier
            </p>

            {opencvError ? (
              <p className="text-red-400 text-sm">{opencvError}</p>
            ) : opencvResult ? (
              <>
                <div className="flex gap-2 mb-4 text-xs">
                  <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-full font-medium mono-value">
                    {opencvResult.processingTime.toFixed(1)}ms
                  </span>
                  <span className="bg-zinc-800 text-zinc-400 border border-zinc-700/50 px-2.5 py-1 rounded-full font-medium">
                    {opencvResult.faces.length} face
                    {opencvResult.faces.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {/* Detection overlay */}
                <div className="canvas-frame mb-4">
                  <canvas
                    ref={opencvCanvasRef}
                    className="w-full h-auto block"
                  />
                </div>

                {/* Cropped faces */}
                {opencvResult.croppedFaces.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
                      Cropped Faces
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
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
                            className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                          >
                            <svg
                              className="h-5 w-5 text-white"
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
                          <span className="absolute top-1 left-1 bg-emerald-500/90 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">
                            #{i + 1}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {opencvResult.faces.length === 0 && (
                  <p className="text-zinc-600 text-center py-6 text-sm">
                    No faces detected
                  </p>
                )}
              </>
            ) : null}
          </div>

          {/* face-api.js WebGL Panel */}
          <div className={`glass-card panel-blue ${faceApiResult ? "has-result" : ""}`}>
            <div className="flex items-center gap-2.5 mb-3">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
              <h2 className="text-base sm:text-lg font-bold text-zinc-200">
                face-api.js
              </h2>
              <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-medium ml-auto">WebGL</span>
            </div>
            <p className="text-xs text-zinc-500 mb-3">
              TinyFaceDetector (Neural Network)
            </p>

            {faceApiError ? (
              <p className="text-red-400 text-sm">{faceApiError}</p>
            ) : faceApiResult ? (
              <>
                <div className="flex gap-2 mb-4 text-xs">
                  <span className="bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2.5 py-1 rounded-full font-medium mono-value">
                    {faceApiResult.processingTime.toFixed(1)}ms
                  </span>
                  <span className="bg-zinc-800 text-zinc-400 border border-zinc-700/50 px-2.5 py-1 rounded-full font-medium">
                    {faceApiResult.faces.length} face
                    {faceApiResult.faces.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {/* Detection overlay */}
                <div className="canvas-frame mb-4">
                  <canvas
                    ref={faceApiCanvasRef}
                    className="w-full h-auto block"
                  />
                </div>

                {/* Cropped faces */}
                {faceApiResult.croppedFaces.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
                      Cropped Faces
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
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
                            className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                          >
                            <svg
                              className="h-5 w-5 text-white"
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
                          <span className="absolute top-1 left-1 bg-blue-500/90 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">
                            #{i + 1}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {faceApiResult.faces.length === 0 && (
                  <p className="text-zinc-600 text-center py-6 text-sm">
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
        <div className="glass-card mb-6 sm:mb-8 animate-fade-in-delay">
          <h2 className="text-base sm:text-lg font-bold text-zinc-200 mb-4 flex items-center gap-2">
            <svg className="h-4 w-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Speed Comparison
          </h2>

          <div className="space-y-4">
            {/* OpenCV bar */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-zinc-400">
                  OpenCV.js <span className="text-zinc-600">(WASM)</span>
                </span>
                <span className="text-xs font-bold text-emerald-400 mono-value">
                  {opencvResult.processingTime.toFixed(1)}ms
                </span>
              </div>
              <div className="w-full bg-zinc-800/80 rounded-lg overflow-hidden">
                <div
                  className="speed-bar bg-gradient-to-r from-emerald-600 to-emerald-500"
                  style={{
                    width:
                      maxTime > 0
                        ? `${Math.max(8, (opencvResult.processingTime / maxTime) * 100)}%`
                        : "8%",
                  }}
                >
                  <span className="text-xs">{opencvResult.faces.length} face
                  {opencvResult.faces.length !== 1 ? "s" : ""}</span>
                </div>
              </div>
            </div>

            {/* face-api.js bar */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-zinc-400">
                  face-api.js <span className="text-zinc-600">(WebGL)</span>
                </span>
                <span className="text-xs font-bold text-blue-400 mono-value">
                  {faceApiResult.processingTime.toFixed(1)}ms
                </span>
              </div>
              <div className="w-full bg-zinc-800/80 rounded-lg overflow-hidden">
                <div
                  className="speed-bar bg-gradient-to-r from-blue-600 to-blue-500"
                  style={{
                    width:
                      maxTime > 0
                        ? `${Math.max(8, (faceApiResult.processingTime / maxTime) * 100)}%`
                        : "8%",
                  }}
                >
                  <span className="text-xs">{faceApiResult.faces.length} face
                  {faceApiResult.faces.length !== 1 ? "s" : ""}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="mt-5 p-3.5 rounded-lg bg-zinc-800/60 border border-zinc-700/40">
            <p className="text-sm text-zinc-300">
              {opencvResult.processingTime < faceApiResult.processingTime ? (
                <>
                  <span className="font-bold text-emerald-400">
                    OpenCV WASM
                  </span>{" "}
                  was{" "}
                  <span className="font-bold mono-value">
                    {(
                      faceApiResult.processingTime /
                      opencvResult.processingTime
                    ).toFixed(1)}
                    x
                  </span>{" "}
                  faster
                </>
              ) : opencvResult.processingTime >
                faceApiResult.processingTime ? (
                <>
                  <span className="font-bold text-blue-400">
                    face-api.js WebGL
                  </span>{" "}
                  was{" "}
                  <span className="font-bold mono-value">
                    {(
                      opencvResult.processingTime /
                      faceApiResult.processingTime
                    ).toFixed(1)}
                    x
                  </span>{" "}
                  faster
                </>
              ) : (
                <>Both methods had the same processing time</>
              )}
            </p>
            <p className="text-[11px] text-zinc-600 mt-1.5">
              First run may be slower due to JIT warmup. Re-run for accurate comparison.
            </p>
          </div>
        </div>
      )}

      {/* Technical Details */}
      <div className="glass-card">
        <h2 className="text-base sm:text-lg font-bold text-zinc-200 mb-4 flex items-center gap-2">
          <svg className="h-4 w-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          Technical Details
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 text-xs sm:text-sm">
          <div className="space-y-2">
            <h3 className="font-semibold text-emerald-400 text-xs uppercase tracking-wider mb-2">
              OpenCV.js — WASM
            </h3>
            <div className="space-y-1.5 text-zinc-400">
              <p>
                <span className="text-zinc-500">Engine:</span>{" "}
                <span className="text-zinc-300">OpenCV 4.9.0 → WebAssembly</span>
              </p>
              <p>
                <span className="text-zinc-500">Algorithm:</span>{" "}
                <span className="text-zinc-300">Haar Cascade Classifier</span>
              </p>
              <p>
                <span className="text-zinc-500">Approach:</span>{" "}
                <span className="text-zinc-300">Classical CV (feature-based)</span>
              </p>
              <p>
                <span className="text-zinc-500">Execution:</span>{" "}
                <span className="text-zinc-300">CPU via WASM</span>
              </p>
            </div>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold text-blue-400 text-xs uppercase tracking-wider mb-2">
              face-api.js — WebGL
            </h3>
            <div className="space-y-1.5 text-zinc-400">
              <p>
                <span className="text-zinc-500">Engine:</span>{" "}
                <span className="text-zinc-300">TensorFlow.js + WebGL</span>
              </p>
              <p>
                <span className="text-zinc-500">Algorithm:</span>{" "}
                <span className="text-zinc-300">TinyFaceDetector (CNN)</span>
              </p>
              <p>
                <span className="text-zinc-500">Approach:</span>{" "}
                <span className="text-zinc-300">Deep learning (neural net)</span>
              </p>
              <p>
                <span className="text-zinc-500">Execution:</span>{" "}
                <span className="text-zinc-300">GPU via WebGL</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center mt-8 pb-4">
        <p className="text-[11px] text-zinc-700">
          All processing runs locally in your browser. No data is sent to any server.
        </p>
      </div>
    </div>
  );
}
