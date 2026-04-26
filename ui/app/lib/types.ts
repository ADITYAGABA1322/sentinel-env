export type ViewMode = "landing" | "mission" | "judge";
export type TaskType = "task1" | "task2" | "task3";
export type ActionType = "delegate" | "verify" | "solve_independently" | "skip";
export type AutoPolicy = "heuristic" | "random" | "trained";

export type Observation = {
  session_id: string;
  scenario_id: string;
  task_type: TaskType;
  difficulty: string;
  task_description: string;
  current_subtask: string;
  subtask_index: number;
  subtasks_total: number;
  subtasks_remaining: number;
  available_specialists: string[];
  available_workers?: string[];
  trust_snapshot: Record<string, number>;
  stakes_level: number;
  step_count: number;
  max_steps: number;
  last_action_summary: string | null;
  last_reward: number;
  episode_status: string;
  gpu_pool?: any[];
};

export type Reward = {
  value: number;
  reason: string;
  signal_breakdown?: Record<string, number>;
};

export type StepResult = {
  observation: Observation;
  reward: Reward;
  done: boolean;
  info: {
    session_id: string;
    step_count: number;
    max_steps: number;
    total_reward: number;
    score: number;
    adversarial_detections?: number;
    adversarial_poisonings?: number;
    environment_mode?: string;
  };
};

export type EvalSummary = {
  avg_score: number;
  avg_completion_rate: number;
  avg_detection_rate: number;
  avg_trust_calibration: number;
  avg_steps: number;
};

export type EvaluationData = {
  summary: {
    random: EvalSummary;
    heuristic: EvalSummary;
    oracle_lite: EvalSummary;
    trained?: EvalSummary;
  };
  by_task: {
    task1: Record<string, EvalSummary>;
    task2: Record<string, EvalSummary>;
    task3: Record<string, EvalSummary>;
  };
};

export type EventItem = {
  step: number;
  action: string;
  specialist?: string;
  summary: string;
  reward: number;
  outcome: "success" | "blocked" | "poisoned" | "skipped" | "reset";
};
