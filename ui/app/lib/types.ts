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
<<<<<<< HEAD
=======
  available_workers?: string[];
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
  trust_snapshot: Record<string, number>;
  stakes_level: number;
  step_count: number;
  max_steps: number;
  last_action_summary: string | null;
  last_reward: number;
  episode_status: string;
<<<<<<< HEAD
=======
  gpu_pool?: any[];
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
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
<<<<<<< HEAD
=======
    environment_mode?: string;
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
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
