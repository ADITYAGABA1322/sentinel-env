"use client";

<<<<<<< HEAD
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
=======
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387

type NodeStatus = "ACTIVE" | "IDLE" | "OVERLOADED" | "FAILED";

interface GPUNode {
  id: string;
  utilization: number;
  memory: number;
  load: number;
  status: NodeStatus;
}

<<<<<<< HEAD
export default function GPUClusterPanel() {
  const [mounted, setMounted] = useState(false);
  const [nodes, setNodes] = useState<GPUNode[]>([
    { id: "GPU-1", utilization: 45, memory: 32, load: 1.2, status: "ACTIVE" },
    { id: "GPU-2", utilization: 12, memory: 8, load: 0.4, status: "IDLE" },
    { id: "GPU-3", utilization: 88, memory: 64, load: 2.8, status: "ACTIVE" },
=======
interface GPUClusterPanelProps {
  sessionId?: string;
  mode?: string;
  gpuPool?: any[]; // Live data from observation
}

export default function GPUClusterPanel({ sessionId, mode, gpuPool }: GPUClusterPanelProps) {
  const [mounted, setMounted] = useState(false);
  const [nodes, setNodes] = useState<GPUNode[]>([
    { id: "GPU-1", utilization: 0, memory: 0, load: 0, status: "IDLE" },
    { id: "GPU-2", utilization: 0, memory: 0, load: 0, status: "IDLE" },
    { id: "GPU-3", utilization: 0, memory: 0, load: 0, status: "IDLE" },
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
    { id: "GPU-4", utilization: 0, memory: 0, load: 0, status: "IDLE" },
  ]);

  const [avgLoad, setAvgLoad] = useState(0);
  const [jitter, setJitter] = useState(0.45);

<<<<<<< HEAD
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
=======
  useEffect(() => { setMounted(true); }, []);

  // ── LIVE SYNC FROM OBSERVATION ────────────────────────────
  useEffect(() => {
    if (gpuPool && Array.isArray(gpuPool)) {
      setNodes(gpuPool.slice(0, 4).map((g: any) => {
        const util = (g.memory_used / g.memory_total) * 100;
        let status = g.state.toUpperCase();
        if (status === "ALLOCATED") status = "ACTIVE";
        
        return {
          id: g.id,
          utilization: util,
          memory: util,
          load: (util / 100) * 4.2,
          status: status as NodeStatus
        };
      }));
    } else if (!sessionId || mode !== "cluster") {
      // Fallback to subtle idle simulation if no live data
      const timer = setInterval(() => {
        setJitter(Math.random() * 0.5);
        setNodes(prev => prev.map(n => ({
          ...n,
          utilization: Math.max(0, n.utilization + (Math.random() - 0.5) * 2),
          load: n.utilization * 0.04
        })));
      }, 2000);
      return () => clearInterval(timer);
    }
  }, [gpuPool, sessionId, mode]);
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387

  useEffect(() => {
    const total = nodes.reduce((acc, n) => acc + n.utilization, 0);
    setAvgLoad(total / nodes.length);
  }, [nodes]);

<<<<<<< HEAD
  if (!mounted) {
    return (
      <section className="section-block" id="gpu-cluster" style={{ opacity: 0 }}>
        <div className="section-label">03 // COMPUTE RESOURCES</div>
        <h2 className="section-title">GPU Compute Clusters</h2>
      </section>
    );
  }
=======
  if (!mounted) return null;
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387

  return (
    <section className="section-block" id="gpu-cluster">
      <div className="section-label">03 // COMPUTE RESOURCES</div>
      <h2 className="section-title">GPU Compute Clusters</h2>
      <p className="section-desc">
<<<<<<< HEAD
        Real-time telemetry from the underlying inference hardware.
        High cluster utilization may introduce latency in the trust calibration loop.
=======
        Real-time telemetry from the underlying inference hardware. 
        Note how cluster utilization spikes as the RL model allocates worker jobs.
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
      </p>

      <div className="cluster-grid">
        {nodes.map((node) => (
          <div key={node.id} className={`card node-card ${node.status.toLowerCase()}`}>
<<<<<<< HEAD
            <div className="card-id">{node.id} // NODE-0{node.id.split("-")[1]}</div>

            <div className="node-status-badge">
              <div className="status-dot" style={{
                background: node.status === "OVERLOADED" ? "var(--red)" :
                  node.status === "FAILED" ? "#555" :
                    node.status === "IDLE" ? "var(--muted)" : "var(--green)"
=======
            <div className="card-id">{node.id} // CORE-AX-{node.id.split("-")[1] || "0X"}</div>
            
            <div className="node-status-badge">
              <div className="status-dot" style={{ 
                background: node.status === "ACTIVE" ? "var(--green)" :
                            node.status === "OVERLOADED" ? "var(--red)" : 
                            node.status === "FAILED" ? "#555" : "var(--muted)" 
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
              }} />
              {node.status}
            </div>

            <div className="metric-bar-wrap" style={{ marginTop: 20 }}>
              <div className="metric-bar-label">
                <span>UTILIZATION</span>
                <span style={{ color: "var(--cyan)" }}>{Math.round(node.utilization)}%</span>
              </div>
              <div className="metric-bar-bg">
<<<<<<< HEAD
                <motion.div
                  className="metric-bar-fill"
                  animate={{ width: `${node.utilization}%` }}
=======
                <motion.div 
                  className="metric-bar-fill" 
                  animate={{ width: `${node.utilization}%` }}
                  transition={{ type: "spring", stiffness: 100, damping: 20 }}
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
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
<<<<<<< HEAD
                <motion.div
                  className="metric-bar-fill"
                  animate={{ width: `${node.memory}%` }}
=======
                <motion.div 
                  className="metric-bar-fill" 
                  animate={{ width: `${node.memory}%` }}
                  transition={{ type: "spring", stiffness: 100, damping: 20 }}
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
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
<<<<<<< HEAD
            <motion.div
=======
            <motion.div 
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
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
