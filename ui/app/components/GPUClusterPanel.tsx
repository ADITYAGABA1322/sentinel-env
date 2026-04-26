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
}

export default function GPUClusterPanel() {
  const [mounted, setMounted] = useState(false);
  const [nodes, setNodes] = useState<GPUNode[]>([
    { id: "GPU-1", utilization: 45, memory: 32, load: 1.2, status: "ACTIVE" },
    { id: "GPU-2", utilization: 12, memory: 8, load: 0.4, status: "IDLE" },
    { id: "GPU-3", utilization: 88, memory: 64, load: 2.8, status: "ACTIVE" },
    { id: "GPU-4", utilization: 0, memory: 0, load: 0, status: "IDLE" },
  ]);

  const [avgLoad, setAvgLoad] = useState(0);
  const [jitter, setJitter] = useState(0.45);

  useEffect(() => {
    setMounted(true);
    const interval = setInterval(() => {
      setJitter(Math.random() * 2);
      setNodes((prev) =>
        prev.map((node) => {
          if (node.status === "FAILED") {
            if (Math.random() > 0.95) return { ...node, status: "IDLE", utilization: 0, load: 0 };
            return node;
          }
          if (Math.random() > 0.995) {
            return { ...node, status: "FAILED", utilization: 0, memory: 0, load: 0 };
          }
          let util = node.utilization + (Math.random() - 0.5) * 15;
          if (Math.random() > 0.9) util += 30;
          util = Math.max(0, Math.min(100, util));
          const mem = Math.max(0, Math.min(100, node.memory + (Math.random() - 0.5) * 5));
          const load = (util / 100) * 4;
          let status: NodeStatus = "ACTIVE";
          if (util > 90) status = "OVERLOADED";
          else if (util < 5) status = "IDLE";
          return { ...node, utilization: util, memory: mem, load, status };
        })
      );
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const total = nodes.reduce((acc, n) => acc + n.utilization, 0);
    setAvgLoad(total / nodes.length);
  }, [nodes]);

  if (!mounted) {
    return (
      <section className="section-block" id="gpu-cluster" style={{ opacity: 0 }}>
        <div className="section-label">03 // COMPUTE RESOURCES</div>
        <h2 className="section-title">GPU Compute Clusters</h2>
      </section>
    );
  }

  return (
    <section className="section-block" id="gpu-cluster">
      <div className="section-label">03 // COMPUTE RESOURCES</div>
      <h2 className="section-title">GPU Compute Clusters</h2>
      <p className="section-desc">
        Real-time telemetry from the underlying inference hardware.
        High cluster utilization may introduce latency in the trust calibration loop.
      </p>

      <div className="cluster-grid">
        {nodes.map((node) => (
          <div key={node.id} className={`card node-card ${node.status.toLowerCase()}`}>
            <div className="card-id">{node.id} // NODE-0{node.id.split("-")[1]}</div>

            <div className="node-status-badge">
              <div className="status-dot" style={{
                background: node.status === "OVERLOADED" ? "var(--red)" :
                  node.status === "FAILED" ? "#555" :
                    node.status === "IDLE" ? "var(--muted)" : "var(--green)"
              }} />
              {node.status}
            </div>

            <div className="metric-bar-wrap" style={{ marginTop: 20 }}>
              <div className="metric-bar-label">
                <span>UTILIZATION</span>
                <span style={{ color: "var(--cyan)" }}>{Math.round(node.utilization)}%</span>
              </div>
              <div className="metric-bar-bg">
                <motion.div
                  className="metric-bar-fill"
                  animate={{ width: `${node.utilization}%` }}
                  style={{ background: node.utilization > 90 ? "var(--red)" : "var(--cyan)" } as any}
                />
              </div>
            </div>

            <div className="metric-bar-wrap" style={{ marginTop: 12 }}>
              <div className="metric-bar-label">
                <span>MEMORY USAGE</span>
                <span style={{ color: "var(--green)" }}>{Math.round(node.memory)}%</span>
              </div>
              <div className="metric-bar-bg">
                <motion.div
                  className="metric-bar-fill"
                  animate={{ width: `${node.memory}%` }}
                  style={{ background: "var(--green)" } as any}
                />
              </div>
            </div>

            <div className="node-footer-stats">
              <div className="node-stat">
                <span className="label">COMPUTE</span>
                <span className="val">{node.load.toFixed(1)} TFLOPS</span>
              </div>
              <div className="node-stat">
                <span className="label">TEMP</span>
                <span className="val">{Math.round(40 + (node.utilization * 0.4))}°C</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="cluster-footer">
        <div className="cluster-total-load">
          <span className="label">TOTAL CLUSTER LOAD</span>
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
          <span>LATENCY: <b>{Math.round(12 + (avgLoad * 0.2))}ms</b></span>
          <span>JITTER: <b>{jitter.toFixed(2)}ms</b></span>
        </div>
      </div>
    </section>
  );
}
