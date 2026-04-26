"use client";

type Props = { running: boolean; done: boolean; adversarialCount: number };

export default function SystemModules({ running, done, adversarialCount }: Props) {
  return (
    <div className="cards-grid">
      <div className="card" style={{ "--card-accent": "#00f5ff" } as React.CSSProperties}>
        <div className="card-id">MOD-001 // ENVIRONMENT</div>
        <div className="card-icon">
          <svg viewBox="0 0 40 40" fill="none"><rect x="2" y="2" width="16" height="16" stroke="#00F5FF" strokeWidth="1" opacity="0.6"/><rect x="22" y="2" width="16" height="16" stroke="#00F5FF" strokeWidth="1" opacity="0.3"/><rect x="2" y="22" width="16" height="16" stroke="#00F5FF" strokeWidth="1" opacity="0.3"/><rect x="22" y="22" width="16" height="16" stroke="#00F5FF" strokeWidth="1" opacity="0.6"/><line x1="18" y1="10" x2="22" y2="10" stroke="#00F5FF" strokeWidth="1"/><line x1="10" y1="18" x2="10" y2="22" stroke="#00F5FF" strokeWidth="1"/><circle cx="10" cy="10" r="2" fill="#00F5FF"/><circle cx="30" cy="30" r="2" fill="#00F5FF" opacity="0.5"/></svg>
        </div>
        <div className="card-title">Multi-Agent Environment</div>
        <div className="card-body">
          Discrete-time partially observable environment hosting N heterogeneous agents.
          Supports configurable adversarial injection ratios and stochastic reward structures per episode.
        </div>
        <div className="card-footer">
          <div className="card-status">
            <div className="status-dot" style={running ? {} : { background: done ? "var(--amber)" : "var(--muted)", animation: "none" }} />
            {running ? "RUNNING" : done ? "COMPLETE" : "IDLE"}
          </div>
          <div className="card-ver">gym v0.26.2</div>
        </div>
      </div>

      <div className="card" style={{ "--card-accent": "#00ff88" } as React.CSSProperties}>
        <div className="card-id">MOD-002 // TRUST ENGINE</div>
        <div className="card-icon">
          <svg viewBox="0 0 40 40" fill="none"><circle cx="20" cy="20" r="14" stroke="#00FF88" strokeWidth="1" strokeDasharray="4 2" opacity="0.5"/><circle cx="20" cy="20" r="8" stroke="#00FF88" strokeWidth="1" opacity="0.8"/><circle cx="20" cy="20" r="3" fill="#00FF88"/><line x1="20" y1="6" x2="20" y2="12" stroke="#00FF88" strokeWidth="1" opacity="0.5"/><line x1="20" y1="28" x2="20" y2="34" stroke="#00FF88" strokeWidth="1" opacity="0.5"/><line x1="6" y1="20" x2="12" y2="20" stroke="#00FF88" strokeWidth="1" opacity="0.5"/><line x1="28" y1="20" x2="34" y2="20" stroke="#00FF88" strokeWidth="1" opacity="0.5"/></svg>
        </div>
        <div className="card-title">Trust Calibration Engine</div>
        <div className="card-body">
          Bayesian trust scoring module that maintains per-agent belief distributions.
          Updates posteriors using observed action-outcome consistency.
        </div>
        <div className="card-footer">
          <div className="card-status">
            <div className="status-dot" />
            CALIBRATING
          </div>
          <div className="card-ver">TCE v1.1.4</div>
        </div>
      </div>

      <div className="card" style={{ "--card-accent": "#ff2d55" } as React.CSSProperties}>
        <div className="card-id">MOD-003 // ADV DETECTION</div>
        <div className="card-icon">
          <svg viewBox="0 0 40 40" fill="none"><polygon points="20,4 36,32 4,32" stroke="#FF2D55" strokeWidth="1" fill="none" opacity="0.7"/><line x1="20" y1="14" x2="20" y2="24" stroke="#FF2D55" strokeWidth="1.5"/><circle cx="20" cy="28" r="1.5" fill="#FF2D55"/></svg>
        </div>
        <div className="card-title">Adversarial Detection Layer</div>
        <div className="card-body">
          Anomaly-based detector using temporal divergence scoring across agent action histories.
          Flags Byzantine agents via KL-divergence threshold on expected vs observed policy distributions.
        </div>
        <div className="card-footer">
          <div className="card-status" style={{ color: adversarialCount > 0 ? "var(--red)" : "var(--green)" }}>
            <div className="status-dot" style={adversarialCount > 0 ? { background: "var(--red)" } : {}} />
            {adversarialCount > 0 ? `${adversarialCount} THREAT${adversarialCount > 1 ? "S" : ""}` : "CLEAR"}
          </div>
          <div className="card-ver">ADL v2.0.1</div>
        </div>
      </div>

      <div className="card" style={{ "--card-accent": "#ffb800" } as React.CSSProperties}>
        <div className="card-id">MOD-004 // RL OPTIMIZER</div>
        <div className="card-icon">
          <svg viewBox="0 0 40 40" fill="none"><polyline points="4,32 10,20 16,26 22,14 28,18 36,6" stroke="#FFB800" strokeWidth="1.5" fill="none"/><circle cx="10" cy="20" r="2" fill="#FFB800" opacity="0.6"/><circle cx="22" cy="14" r="2" fill="#FFB800" opacity="0.6"/><circle cx="36" cy="6" r="2.5" fill="#FFB800"/><line x1="4" y1="35" x2="36" y2="35" stroke="#FFB800" strokeWidth="0.5" opacity="0.3"/><line x1="4" y1="35" x2="4" y2="4" stroke="#FFB800" strokeWidth="0.5" opacity="0.3"/></svg>
        </div>
        <div className="card-title">Reinforcement Learning Optimizer</div>
        <div className="card-body">
          Proximal Policy Optimization (PPO) with trust-weighted reward shaping.
          Policy gradient updates incorporate adversarial penalty terms.
        </div>
        <div className="card-footer">
          <div className="card-status" style={{ color: "var(--amber)" }}>
            <div className="status-dot" style={{ background: "var(--amber)" }} />
            TRAINING
          </div>
          <div className="card-ver">PPO v3.2.0</div>
        </div>
      </div>
    </div>
  );
}
