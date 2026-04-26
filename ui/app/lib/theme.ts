/* ── Design tokens & helpers ────────────────────────────── */

export const colors = {
  trust: {
    high:    "#34d399",
    medium:  "#fbbf24",
    low:     "#f43f5e",
    neutral: "#818cf8",
  },
  accent:    "#818cf8",
  accentAlt: "#c084fc",
  success:   "#34d399",
  danger:    "#f43f5e",
  warning:   "#f59e0b",
} as const;

export function trustColor(v: number): string {
  if (v >= 0.72) return colors.trust.high;
  if (v >= 0.5)  return colors.trust.neutral;
  if (v >= 0.3)  return colors.trust.medium;
  return colors.trust.low;
}

export function trustLabel(v: number): string {
  if (v >= 0.72) return "Trusted";
  if (v >= 0.5)  return "Neutral";
  if (v >= 0.3)  return "Uncertain";
  return "Suspicious";
}

export function formatScore(v: number | undefined): string {
  return typeof v === "number" ? v.toFixed(3) : "—";
}

export const spring  = { type: "spring" as const, stiffness: 300, damping: 30 };
export const gentle  = { type: "spring" as const, stiffness: 200, damping: 25 };
