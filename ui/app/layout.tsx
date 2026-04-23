import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SENTINEL",
  description:
    "Trust-calibration environment for long-horizon multi-agent RL training."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
