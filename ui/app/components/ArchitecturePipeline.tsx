"use client";

export default function ArchitecturePipeline() {
  const nodes = [
    { id: "LAYER-01", name: "AGENTS", desc: "S0–S4 emit\nobservations + actions\nper timestep", color: "rgba(0, 200, 255, 0.4)", labelColor: "#00f5ff" },
    { id: "LAYER-02", name: "ADV DETECTOR", desc: "KL-divergence\nanomaly scan\nByzantine flag", color: "rgba(255, 45, 85, 0.3)", labelColor: "#ff2d55" },
    { id: "LAYER-03", name: "ORCHESTRATOR", desc: "Trust-weighted\naggregation &\ndecision output", color: "rgba(0, 255, 136, 0.3)", labelColor: "#00ff88" },
    { id: "LAYER-04", name: "REWARD SIG.", desc: "Shaped scalar\nwith adversarial\npenalty term", color: "rgba(255, 184, 0, 0.3)", labelColor: "#ffb800" },
    { id: "LAYER-05", name: "POLICY UPDATE", desc: "PPO gradient\nstep + trust\nposterior update", color: "rgba(0, 85, 255, 0.4)", labelColor: "#0088ff" },
  ];

  const arrows = ["ACTIONS", "FLAGS", "DECISION", "REWARD"];

  return (
    <>
      <div className="arch-flow">
        {nodes.map((node, i) => (
          <span key={node.id} style={{ display: "contents" }}>
            <div className="arch-node" style={{ "--node-color": node.color } as React.CSSProperties}>
              <div className="arch-node-id" style={{ color: node.labelColor }}>{node.id}</div>
              <div className="arch-node-name">{node.name}</div>
              <div className="arch-node-desc">{node.desc.split("\n").map((l, j) => <span key={j}>{l}<br /></span>)}</div>
            </div>
            {i < nodes.length - 1 && (
              <div className="arch-arrow">
                <div className="arch-arrow-line" />
                <div className="arch-arrow-label">{arrows[i]}</div>
              </div>
            )}
          </span>
        ))}
      </div>

      <div className="arch-code-loop">
        <span style={{ color: "var(--muted)" }}>LOOP:</span>&nbsp;&nbsp;
        <span style={{ color: "var(--cyan)" }}>observe()</span> →{" "}
        <span style={{ color: "var(--red)" }}>detect_adversary()</span> →{" "}
        <span style={{ color: "var(--green)" }}>aggregate_trust()</span> →{" "}
        <span style={{ color: "var(--white)" }}>act()</span> →{" "}
        <span style={{ color: "var(--amber)" }}>compute_reward()</span> →{" "}
        <span style={{ color: "#0088ff" }}>update_policy()</span> →{" "}
        <span style={{ color: "var(--cyan)" }}>repeat</span>
        &nbsp;&nbsp;
        <span style={{ color: "var(--muted)" }}>// T: O(N·K) // SPACE: O(N²)</span>
      </div>
    </>
  );
}
