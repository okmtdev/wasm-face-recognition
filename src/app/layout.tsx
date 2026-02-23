import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Face Detection — WASM vs WebGL",
  description:
    "Compare face detection performance between OpenCV.js (WASM) and face-api.js (WebGL)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-grid antialiased">{children}</body>
    </html>
  );
}
