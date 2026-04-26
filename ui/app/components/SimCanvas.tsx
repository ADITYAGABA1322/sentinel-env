"use client";

import { useRef, useEffect } from "react";

type AgentData = { id: string; trust: number; type: string };

type Props = {
  trustSnapshot: Record<string, number>;
  adversarialAgents: Set<string>;
  activeSpec: string | null;
};

const AGENT_ANGLES = [0, 72, 144, 216, 288];
const COLORS: Record<string, string> = {
  orch: "#00F5FF", normal: "#00FF88", degraded: "#FFB800", adversarial: "#FF2D55",
};

function hexToRgb(hex: string) {
  if (hex === "#00F5FF") return "0,245,255";
  if (hex === "#00FF88") return "0,255,136";
  if (hex === "#FFB800") return "255,184,0";
  if (hex === "#FF2D55") return "255,45,85";
  return "0,200,255";
}

function agentType(id: string, trust: number, isAdv: boolean): string {
  if (isAdv) return "adversarial";
  if (trust < 0.4) return "degraded";
  return "normal";
}

export default function SimCanvas({ trustSnapshot, adversarialAgents, activeSpec }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const dataRef = useRef({ trustSnapshot, adversarialAgents, activeSpec });

  useEffect(() => {
    dataRef.current = { trustSnapshot, adversarialAgents, activeSpec };
  }, [trustSnapshot, adversarialAgents, activeSpec]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 0, H = 0, tick = 0, animId = 0;
    type Packet = { agentIdx: number; progress: number; dir: number };
    let dataPackets: Packet[] = [];

    function resize() {
      W = canvas!.width = canvas!.offsetWidth;
      H = canvas!.height = canvas!.offsetHeight;
    }

    function getPos(angle: number, cx: number, cy: number, r: number) {
      const rad = ((angle - 90) * Math.PI) / 180;
      return { x: cx + Math.cos(rad) * r, y: cy + Math.sin(rad) * r };
    }

    function draw() {
      const { trustSnapshot: ts, adversarialAgents: advSet, activeSpec: active } = dataRef.current;
      ctx!.clearRect(0, 0, W, H);
      tick++;

      const cx = W / 2, cy = H / 2;
      const outerR = Math.min(W, H) * 0.35;
      const orchR = 30, agentR = 18;

      const agents: AgentData[] = ["S0", "S1", "S2", "S3", "S4"].map((id, i) => ({
        id, trust: ts[id] ?? 0.5,
        type: i === 0 ? "orch" : agentType(id, ts[id] ?? 0.5, advSet.has(id)),
      }));

      // Grid
      ctx!.strokeStyle = "rgba(0,100,200,0.04)";
      ctx!.lineWidth = 0.5;
      for (let x = 0; x < W; x += 40) { ctx!.beginPath(); ctx!.moveTo(x, 0); ctx!.lineTo(x, H); ctx!.stroke(); }
      for (let y = 0; y < H; y += 40) { ctx!.beginPath(); ctx!.moveTo(0, y); ctx!.lineTo(W, y); ctx!.stroke(); }

      // Orbit ring
      ctx!.beginPath(); ctx!.arc(cx, cy, outerR, 0, Math.PI * 2);
      ctx!.strokeStyle = "rgba(0,200,255,0.06)"; ctx!.lineWidth = 1;
      ctx!.setLineDash([4, 8]); ctx!.stroke(); ctx!.setLineDash([]);

      // Scanning ring
      const scanAngle = (tick * 0.02) % (Math.PI * 2);
      ctx!.save(); ctx!.translate(cx, cy); ctx!.rotate(scanAngle);
      const scanArc = ctx!.createLinearGradient(-outerR, 0, outerR, 0);
      scanArc.addColorStop(0, "rgba(0,200,255,0)");
      scanArc.addColorStop(1, "rgba(0,200,255,0.08)");
      ctx!.beginPath(); ctx!.moveTo(0, 0); ctx!.arc(0, 0, outerR, -0.4, 0); ctx!.closePath();
      ctx!.fillStyle = scanArc; ctx!.fill(); ctx!.restore();

      // Connections from orch to agents
      const orchPos = { x: cx, y: cy };
      agents.slice(1).forEach((a, i) => {
        const pos = getPos(AGENT_ANGLES[i + 1], cx, cy, outerR);
        const col = COLORS[a.type] || COLORS.normal;
        const alpha = a.type === "adversarial" ? 0.12 : a.trust * 0.35;
        const isDash = a.type === "adversarial";
        if (isDash) ctx!.setLineDash([4, 6]); else ctx!.setLineDash([]);
        ctx!.beginPath(); ctx!.moveTo(orchPos.x, orchPos.y); ctx!.lineTo(pos.x, pos.y);
        ctx!.strokeStyle = `rgba(${hexToRgb(col)},${alpha})`;
        ctx!.lineWidth = a.trust * 2; ctx!.stroke(); ctx!.setLineDash([]);
      });

      // Data packets
      if (tick % 20 === 0) {
        const ai = Math.floor(Math.random() * 4) + 1;
        dataPackets.push({ agentIdx: ai, progress: 0, dir: Math.random() > 0.5 ? 1 : -1 });
      }
      dataPackets = dataPackets.filter(p => p.progress <= 1);
      dataPackets.forEach(p => {
        p.progress += 0.025;
        const a = agents[p.agentIdx];
        const pos = getPos(AGENT_ANGLES[p.agentIdx], cx, cy, outerR);
        const t = p.progress;
        const px = orchPos.x + (pos.x - orchPos.x) * (p.dir > 0 ? t : 1 - t);
        const py = orchPos.y + (pos.y - orchPos.y) * (p.dir > 0 ? t : 1 - t);
        const col = COLORS[a.type] || COLORS.normal;
        ctx!.beginPath(); ctx!.arc(px, py, 3, 0, Math.PI * 2); ctx!.fillStyle = col; ctx!.fill();
        const g = ctx!.createRadialGradient(px, py, 0, px, py, 8);
        g.addColorStop(0, col); g.addColorStop(1, "transparent");
        ctx!.beginPath(); ctx!.arc(px, py, 8, 0, Math.PI * 2); ctx!.fillStyle = g; ctx!.fill();
      });

      // Outer agents
      agents.slice(1).forEach((a, i) => {
        const pos = getPos(AGENT_ANGLES[i + 1], cx, cy, outerR);
        const col = COLORS[a.type] || COLORS.normal;
        const pulse = 0.7 + 0.3 * Math.sin(tick * 0.05 + AGENT_ANGLES[i + 1]);
        const isActive = active === a.id;

        // Glow
        const g = ctx!.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, agentR * 2.5);
        g.addColorStop(0, `rgba(${hexToRgb(col)},${(isActive ? 0.35 : 0.2) * pulse})`);
        g.addColorStop(1, "transparent");
        ctx!.beginPath(); ctx!.arc(pos.x, pos.y, agentR * 2.5, 0, Math.PI * 2);
        ctx!.fillStyle = g; ctx!.fill();

        // Node circle
        ctx!.beginPath(); ctx!.arc(pos.x, pos.y, agentR, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${hexToRgb(col)},0.1)`; ctx!.fill();
        ctx!.strokeStyle = `rgba(${hexToRgb(col)},${0.6 * pulse})`;
        ctx!.lineWidth = isActive ? 2.5 : 1.5; ctx!.stroke();

        // Adversarial warning ring
        if (a.type === "adversarial") {
          ctx!.beginPath();
          ctx!.arc(pos.x, pos.y, agentR + 6 + Math.sin(tick * 0.1) * 3, 0, Math.PI * 2);
          ctx!.strokeStyle = `rgba(255,45,85,${0.3 * pulse})`;
          ctx!.lineWidth = 1; ctx!.setLineDash([3, 4]); ctx!.stroke(); ctx!.setLineDash([]);
        }

        // Labels
        ctx!.font = '9px "Share Tech Mono"'; ctx!.fillStyle = col; ctx!.textAlign = "center";
        ctx!.fillText(a.id, pos.x, pos.y - agentR - 8);
        ctx!.fillStyle = "rgba(232,244,255,0.3)";
        ctx!.fillText(a.trust.toFixed(2), pos.x, pos.y + 4);
      });

      // Orchestrator
      const orchPulse = 0.7 + 0.3 * Math.sin(tick * 0.04);
      const orchG = ctx!.createRadialGradient(cx, cy, 0, cx, cy, orchR * 3);
      orchG.addColorStop(0, `rgba(0,245,255,${0.15 * orchPulse})`);
      orchG.addColorStop(1, "transparent");
      ctx!.beginPath(); ctx!.arc(cx, cy, orchR * 3, 0, Math.PI * 2);
      ctx!.fillStyle = orchG; ctx!.fill();

      ctx!.beginPath(); ctx!.arc(cx, cy, orchR, 0, Math.PI * 2);
      ctx!.fillStyle = "rgba(0,245,255,0.1)"; ctx!.fill();
      ctx!.strokeStyle = `rgba(0,245,255,${0.8 * orchPulse})`;
      ctx!.lineWidth = 2; ctx!.stroke();

      ctx!.beginPath(); ctx!.arc(cx, cy, orchR * 0.6, 0, Math.PI * 2);
      ctx!.strokeStyle = `rgba(0,245,255,${0.4 * orchPulse})`;
      ctx!.lineWidth = 1; ctx!.stroke();

      ctx!.font = 'bold 9px "Share Tech Mono"'; ctx!.fillStyle = "#00F5FF";
      ctx!.textAlign = "center"; ctx!.fillText("S0", cx, cy - 4);
      ctx!.font = '8px "Share Tech Mono"'; ctx!.fillStyle = "rgba(0,245,255,0.5)";
      ctx!.fillText("ORCH", cx, cy + 8);

      animId = requestAnimationFrame(draw);
    }

    const ro = new ResizeObserver(() => resize());
    ro.observe(canvas.parentElement!);
    resize(); draw();

    return () => { cancelAnimationFrame(animId); ro.disconnect(); };
  }, []);

  return <canvas ref={ref} />;
}
