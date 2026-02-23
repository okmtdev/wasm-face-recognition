# wasm-face-recognition

ブラウザ上でWASMとWebGLを使った顔検出を行い、顔部分を切り取るNext.jsアプリケーション。
2つの手法の処理速度を比較できます。

## 手法

| 手法 | エンジン | アルゴリズム | 実行環境 |
|------|---------|-------------|---------|
| **OpenCV.js (WASM)** | OpenCV 4.9.0 → WebAssembly | Haar Cascade Classifier | CPU (WASM) |
| **face-api.js (WebGL)** | TensorFlow.js + WebGL | TinyFaceDetector (CNN) | GPU (WebGL) |

## 機能

- 画像のドラッグ＆ドロップ / ファイル選択アップロード
- 複数人の顔を同時検出
- 検出された顔部分を自動切り取り（ダウンロード可能）
- 2手法の処理速度をリアルタイム比較

## セットアップ

```bash
npm install
npm run dev
```

ブラウザで http://localhost:3000 を開いてください。

OpenCV.js と face-api.js のモデルはブラウザ上でCDNから自動的にロードされます。
初回ロードはライブラリのダウンロードに時間がかかります。

## 技術構成

- **Next.js 14** (App Router, TypeScript)
- **Tailwind CSS** (スタイリング)
- **OpenCV.js** (CDNからWASMバイナリをロード)
- **face-api.js** (@vladmandic/face-api, CDNからモデルをロード)
