import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SatQuery AI — Remote Sensing Intelligence",
  description: "Agentic Vision-Language Assistant for Multimodal Remote Sensing — SIH26167",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#f8f9fb] text-zinc-900 min-h-screen grid-bg">{children}</body>
    </html>
  );
}
