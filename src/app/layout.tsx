import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Face Detection - WASM vs WebGL",
  description:
    "Compare face detection performance between OpenCV.js (WASM) and face-api.js (WebGL)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
