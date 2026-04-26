import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SENTINEL — Multi-Agent Trust Calibration",
  description:
    "Train an orchestrator to decide who to trust, when to verify, and how to recover in long multi-agent tasks under adversarial pressure.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
