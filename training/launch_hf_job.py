from __future__ import annotations

import argparse
import os
import shlex
import sys
from textwrap import dedent

from huggingface_hub import run_job


DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DEFAULT_REPO = "https://github.com/ADITYAGABA1322/sentinel-env"
DEFAULT_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"


def shell_join(lines: list[str]) -> str:
    return " && ".join(line.strip() for line in lines if line.strip())


def bootstrap_repo(repo_url: str) -> list[str]:
    return [
        "set -eux",
        "command -v git || (apt-get update && apt-get install -y git)",
        f"git clone {shlex.quote(repo_url)} sentinel-env",
        "cd sentinel-env",
        "python -m pip install --upgrade pip",
        "pip install -r requirements.txt",
        "pip install -r requirements-train.txt",
    ]


def gpu_test_command() -> str:
    return "python -c 'import torch; print(torch.cuda.get_device_name())'"


def train_command(args: argparse.Namespace) -> str:
    lines = bootstrap_repo(args.repo_url)
    lines.append(
        " ".join(
            [
                "python training/train.py",
                f"--episodes {args.episodes}",
                f"--task {shlex.quote(args.task)}",
                f"--seed {args.seed}",
                f"--model {shlex.quote(args.model)}",
                f"--epochs {args.epochs}",
                f"--batch-size {args.batch_size}",
                f"--learning-rate {args.learning_rate}",
                f"--lora-rank {args.lora_rank}",
                f"--num-generations {args.num_generations}",
                f"--max-seq-length {args.max_seq_length}",
                f"--output-dir {shlex.quote(args.output_dir)}",
            ]
        )
    )
    if args.mode == "train-full":
        upload_code = (
            "import os; "
            "from huggingface_hub import HfApi; "
            "token=os.environ.get('HF_TOKEN'); "
            "api=HfApi(token=token); "
            "model_repo=os.environ.get('SENTINEL_MODEL_REPO','XcodeAddy/sentinel-grpo-qwen05'); "
            "artifact_repo=os.environ.get('SENTINEL_ARTIFACT_REPO','XcodeAddy/sentinel-env-artifacts'); "
            "job_id=os.environ.get('JOB_ID','manual'); "
            "api.create_repo(model_repo, repo_type='model', exist_ok=True); "
            f"api.upload_folder(folder_path='{args.output_dir}', repo_id=model_repo, repo_type='model'); "
            "api.create_repo(artifact_repo, repo_type='dataset', exist_ok=True); "
            "api.upload_folder(folder_path='outputs', repo_id=artifact_repo, repo_type='dataset', path_in_repo=f'job-{job_id}/outputs'); "
            "print('Uploaded model adapter to', model_repo); "
            "print('Uploaded outputs to', artifact_repo, 'under', f'job-{job_id}/outputs')"
        )
        lines.extend(
            [
                "python -c \"from training.replay import record_trained_actions; "
                f"record_trained_actions(adapter_path='{args.output_dir}', "
                f"base_model='{args.model}', tasks=['task1','task2','task3'], "
                "seeds=range(30), out_path='outputs/trained_policy_replay.jsonl')\"",
                "python training/evaluate.py --episodes 30 --task all "
                "--policies random,heuristic,oracle_lite,trained "
                "--replay outputs/trained_policy_replay.jsonl "
                "--out outputs/eval_post.json --no-plot",
                "cp outputs/eval_post.json outputs/evaluation_results.json",
                "python -m training.plots --pre outputs/eval_pre.json "
                "--post outputs/eval_post.json --out-dir outputs/charts",
                f"python -c {shlex.quote(upload_code)}",
            ]
        )
    return shell_join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SENTINEL training on Hugging Face Jobs without shell quoting pain."
    )
    parser.add_argument("--mode", choices=["gpu-test", "train-smoke", "train-full"], default="gpu-test")
    parser.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", "XcodeAddy"))
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--output-dir", default="training/sentinel_qwen05_grpo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            dedent(
                """
                HF_TOKEN is not set.

                Run:
                  read -s HF_TOKEN
                  export HF_TOKEN
                Then paste your Hugging Face write token.
                """
            ).strip()
        )

    command = gpu_test_command() if args.mode == "gpu-test" else train_command(args)
    print("Launching HF Job:")
    print(f"  mode      = {args.mode}")
    print(f"  namespace = {args.namespace}")
    print(f"  flavor    = {args.flavor}")
    print(f"  timeout   = {args.timeout}")
    print(f"  image     = {args.image}")
    print("  command   = bash -lc", shlex.quote(command[:260] + ("..." if len(command) > 260 else "")))

    job = run_job(
        image=args.image,
        command=["bash", "-lc", command],
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        token=token,
        secrets={"HF_TOKEN": token},
        env={
            "SENTINEL_MODEL_REPO": "XcodeAddy/sentinel-grpo-qwen05",
            "SENTINEL_ARTIFACT_REPO": "XcodeAddy/sentinel-env-artifacts",
        },
        labels={"project": "sentinel", "mode": args.mode},
    )
    print("Job launched.")
    print("URL:", job.url)
    print("ID:", job.id)
    print()
    print("Follow logs with:")
    print(f"  .venv/bin/hf jobs logs -f {job.id} --namespace {args.namespace} --token \"$HF_TOKEN\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
