"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

type NodeStatus = "ACTIVE" | "IDLE" | "OVERLOADED" | "FAILED";

interface GPUNode {
  id: string;
  utilization: number;
  memory: number;
  load: number;
  status: NodeStatus;
  temp: number;
}

export default function GPUClusterPanel() {
  const [mounted, setMounted] = useState(false);
  const [nodes, setNodes] = useState<GPUNode[]>([
    { id: "GPU-1", utilization: 45, memory: 32, load: 1.2, status: "ACTIVE", temp: 55 },
    { id: "GPU-2", utilization: 12, memory: 8, load: 0.4, status: "IDLE", temp: 42 },
    { id: "GPU-3", utilization: 88, memory: 64, load: 2.8, status: "ACTIVE", temp: 78 },
    { id: "GPU-4", utilization: 0, memory: 0, load: 0, status: "IDLE", temp: 35 },
  ]);

  const [avgLoad, setAvgLoad] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    const interval = setInterval(() => {
      setNodes((prev) =>
        prev.map((node) => {
          if (node.status === "FAILED") {
            if (Math.random() > 0.95) {
              addLog(`[RECOVERY] ${node.id} initialized. Performing self-test...`);
              return { ...node, status: "IDLE", utilization: 0, load: 0 };
            }
            return node;
          }

          if (Math.random() > 0.995) {
            addLog(`[CRITICAL] ${node.id} core voltage failure! Node offline.`);
            return { ...node, status: "FAILED", utilization: 0, memory: 0, load: 0, temp: 20 };
          }

          let util = node.utilization + (Math.random() - 0.5) * 15;
          if (Math.random() > 0.9) {
            util += 35;
            addLog(`[SPIKE] Massive compute load detected on ${node.id}.`);
          }
          util = Math.max(0, Math.min(100, util));
          
          const mem = Math.max(0, Math.min(100, node.memory + (Math.random() - 0.5) * 8));
          const load = (util / 100) * 4.2;
          const temp = 35 + (util * 0.5) + (Math.random() * 2);
          
          let status: NodeStatus = "ACTIVE";
          if (util > 92) {
            status = "OVERLOADED";
            if (node.status !== "OVERLOADED") addLog(`[WARNING] ${node.id} thermal throttling active.`);
          }
          else if (util < 5) status = "IDLE";

          return { ...node, utilization: util, memory: mem, load, status, temp };
        })
      );
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const total = nodes.reduce((acc, n) => acc + n.utilization, 0);
    setAvgLoad(total / nodes.length);
  }, [nodes]);

  const addLog = (msg: string) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogs(prev => [`[${time}] ${msg}`, ...prev].slice(0, 50));
  };

  if (!mounted) return null;

  return (
    <section className="section-block crazy-gpu" id="gpu-cluster">
      <div className="section-label">03 // COMPUTATIONAL SUBSTRATE</div>
      <h2 className="section-title">Nvidia H100 Cluster Telemetry</h2>
      <p className="section-desc">
        High-fidelity hardware monitoring of the underlying neural inference cluster. 
        Saturation of these nodes directly impacts trust re-calibration latency.
      </p>

      <div className="gpu-layout">
        {/* LEFT: NODE GRID */}
        <div className="gpu-grid-side">
          <div className="cluster-grid">
            {nodes.map((node) => (
              <div key={node.id} className={`card node-card ${node.status.toLowerCase()} crazy-card`}>
                <div className="node-glitch-bg" />
                <div className="card-id">{node.id} // CORE-AX-{node.id.split("-")[1]}</div>
                
                <div className="node-status-badge">
                  <div className="status-dot" style={{ 
                    background: node.status === "OVERLOADED" ? "var(--red)" : 
                                node.status === "FAILED" ? "#555" : 
                                node.status === "IDLE" ? "var(--muted)" : "var(--green)" 
                  }} />
                  {node.status}
                </div>

                {/* VISUAL METER */}
                <div className="node-visual">
                  <svg viewBox="0 0 100 100" className="radial-meter">
                    <circle cx="50" cy="50" r="45" className="meter-bg" />
                    <motion.circle 
                      cx="50" cy="50" r="45" 
                      className="meter-fill"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: node.utilization / 100 }}
                      style={{ stroke: node.utilization > 90 ? "var(--red)" : "var(--cyan)" }}
                    />
                    <text x="50" y="55" className="meter-text">{Math.round(node.utilization)}%</text>
                  </svg>
                </div>

                <div className="node-metrics-stack">
                  <div className="mini-metric">
                    <span className="l">MEM</span>
                    <div className="mini-bar-bg"><motion.div className="mini-bar-fill" animate={{ width: `${node.memory}%` }} /></div>
                  </div>
                  <div className="mini-metric">
                    <span className="l">TMP</span>
                    <div className="mini-bar-bg"><motion.div className="mini-bar-fill tm" animate={{ width: `${(node.temp / 100) * 100}%` }} /></div>
                  </div>
                </div>

                <div className="node-footer-stats">
                  <div className="node-stat">
                    <span className="label">LOAD</span>
                    <span className="val">{node.load.toFixed(1)} TFLOPS</span>
                  </div>
                  <div className="node-stat">
                    <span className="label">FREQ</span>
                    <span className="val">{node.status === "FAILED" ? 0 : (2.4 + (node.utilization * 0.01)).toFixed(2)} GHz</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT: SYSTEM LOG & HEATMAP */}
        <div className="gpu-sys-side">
          <div className="card sys-card">
            <div className="card-id">SYS-LOG // KERNEL TELEMETRY</div>
            <div className="terminal-log" ref={logRef}>
              <AnimatePresence initial={false}>
                {logs.map((log, i) => (
                  <motion.div 
                    key={log + i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="log-line"
                  >
                    {log}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          <div className="card sys-card heatmap-card">
            <div className="card-id">THERMAL // HEATMAP</div>
            <div className="heatmap-grid">
              {Array.from({ length: 64 }).map((_, i) => (
                <motion.div 
                  key={i} 
                  className="heat-cell" 
                  animate={{ 
                    opacity: 0.2 + (Math.random() * 0.8),
                    background: i % 8 < 4 ? "var(--cyan)" : "var(--blue)"
                  }}
                  transition={{ repeat: Infinity, duration: 1 + Math.random() * 2, repeatType: "mirror" }}
                />
              ))}
            </div>
            <div className="heatmap-overlay">SCANNING...</div>
          </div>
        </div>
      </div>

      <div className="cluster-footer crazy-footer">
        <div className="cluster-total-load">
          <span className="label">AGGREGATE CLUSTER PRESSURE</span>
          <div className="load-meter-bg">
            <motion.div 
              className="load-meter-fill"
              animate={{ width: `${avgLoad}%` }}
              style={{ background: avgLoad > 80 ? "var(--red)" : "var(--cyan)", color: avgLoad > 80 ? "var(--red)" : "var(--cyan)" } as any}
            />
          </div>
          <span className="val">{Math.round(avgLoad)}%</span>
        </div>
        <div className="cluster-telemetry">
          <span>THROUGHPUT: <b>{Math.round(140 - (avgLoad * 0.5))} FPS</b></span>
          <span>SYSTEM HEALTH: <b style={{ color: avgLoad > 90 ? "var(--red)" : "var(--green)" }}>{avgLoad > 90 ? "CRITICAL" : "OPTIMAL"}</b></span>
        </div>
      </div>

    </section>
  );
}
