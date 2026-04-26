"use client";

import { useRef, useEffect } from "react";

type Node = {
  x: number; y: number;
  vx: number; vy: number;
  r: number; pulse: number;
  type: "blue" | "green" | "red";
};

export default function HeroCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 0, H = 0;
    let nodes: Node[] = [];
    let animId = 0;

    function resize() {
      W = canvas!.width = canvas!.offsetWidth;
      H = canvas!.height = canvas!.offsetHeight;
      buildNodes();
    }

    function buildNodes() {
      nodes = [];
      const count = Math.floor((W * H) / 18000);
      for (let i = 0; i < count; i++) {
        nodes.push({
          x: Math.random() * W, y: Math.random() * H,
          vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
          r: Math.random() * 2 + 1,
          pulse: Math.random() * Math.PI * 2,
          type: Math.random() < 0.1 ? "red" : Math.random() < 0.15 ? "green" : "blue",
        });
      }
    }

    function draw() {
      ctx!.clearRect(0, 0, W, H);

      // Grid
      ctx!.strokeStyle = "rgba(0,100,200,0.06)";
      ctx!.lineWidth = 0.5;
      for (let x = 0; x < W; x += 60) {
        ctx!.beginPath(); ctx!.moveTo(x, 0); ctx!.lineTo(x, H); ctx!.stroke();
      }
      for (let y = 0; y < H; y += 60) {
        ctx!.beginPath(); ctx!.moveTo(0, y); ctx!.lineTo(W, y); ctx!.stroke();
      }

      // Update
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > W) n.vx *= -1;
        if (n.y < 0 || n.y > H) n.vy *= -1;
        n.pulse += 0.02;
      });

      // Edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 140) {
            const alpha = (1 - dist / 140) * 0.25;
            let color: string;
            if (a.type === "red" || b.type === "red") color = `rgba(255,45,85,${alpha * 0.8})`;
            else if (a.type === "green" || b.type === "green") color = `rgba(0,255,136,${alpha * 0.6})`;
            else color = `rgba(0,200,255,${alpha})`;
            ctx!.strokeStyle = color;
            ctx!.lineWidth = 0.5;
            ctx!.beginPath(); ctx!.moveTo(a.x, a.y); ctx!.lineTo(b.x, b.y); ctx!.stroke();
          }
        }
      }

      // Nodes
      nodes.forEach(n => {
        const pulse = 0.5 + 0.5 * Math.sin(n.pulse);
        let color: string, gcolor: string;
        if (n.type === "red") { color = `rgba(255,45,85,${0.4 + 0.5 * pulse})`; gcolor = "#FF2D55"; }
        else if (n.type === "green") { color = `rgba(0,255,136,${0.4 + 0.5 * pulse})`; gcolor = "#00FF88"; }
        else { color = `rgba(0,200,255,${0.35 + 0.4 * pulse})`; gcolor = "#00F5FF"; }

        const g = ctx!.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 3);
        g.addColorStop(0, gcolor); g.addColorStop(1, "transparent");
        ctx!.fillStyle = g;
        ctx!.beginPath(); ctx!.arc(n.x, n.y, n.r * 3, 0, Math.PI * 2); ctx!.fill();
        ctx!.fillStyle = color;
        ctx!.beginPath(); ctx!.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx!.fill();
      });

      animId = requestAnimationFrame(draw);
    }

    const onResize = () => { cancelAnimationFrame(animId); resize(); draw(); };
    window.addEventListener("resize", onResize);
    resize();
    draw();

    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", onResize); };
  }, []);

  return (
    <div className="hero-canvas-wrap">
      <canvas ref={ref} />
    </div>
  );
}
