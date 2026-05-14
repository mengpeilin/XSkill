from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from xskill.common.replay_buffer import ReplayBuffer


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_manifest_root() -> Path:
    return repo_root() / "zarr"


def default_output_root() -> Path:
    return repo_root() / "experiment" / "realworld_training_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    # parser.add_argument("--human-counts", nargs="+", type=int, default=[40, 70, 100, 200])
    # parser.add_argument("--robot-counts", nargs="+", type=int, default=[5, 40])
    parser.add_argument("--human-counts", nargs="+", type=int, default=[40, 70])
    parser.add_argument("--robot-counts", nargs="+", type=int, default=[40])
    parser.add_argument("--manifest-root", type=Path, default=default_manifest_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Device list used for scheduling. If --max-concurrent exceeds this list, devices are reused in round-robin order.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Total number of jobs to run at once across all listed devices.",
    )
    parser.add_argument("--skill-ckpt", type=int, default=499)
    parser.add_argument("--skill-max-epochs", type=int, default=None)
    parser.add_argument("--skill-save-every", type=int, default=None)
    parser.add_argument("--bc-num-epochs", type=int, default=None)
    parser.add_argument("--bc-ckpt-frequency", type=int, default=None)
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class Job:
    task: str
    human_demos: int
    robot_demos: int
    human_mask: Path
    robot_mask: Path
    human_zarr: Path
    robot_zarr: Path

    @property
    def name(self) -> str:
        return f"{self.task}:human_{self.human_demos}:robot_{self.robot_demos}"


class StageFailure(RuntimeError):
    def __init__(self, stage: str, log_path: Path, returncode: int):
        super().__init__(f"{stage} failed with exit code {returncode}. See {log_path}")
        self.stage = stage
        self.log_path = log_path
        self.returncode = returncode


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as file:
        json.dump(payload, file, indent=2, default=str)
    temp_path.replace(path)


def read_json(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def hydra_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def hydra_list(values) -> str:
    return "[" + ",".join(hydra_scalar(value) for value in values) + "]"


def available_tasks(manifest_root: Path) -> list[str]:
    if not manifest_root.is_dir():
        raise FileNotFoundError(f"Missing manifest root {manifest_root}")
    tasks = sorted(
        path.name
        for path in manifest_root.iterdir()
        if path.is_dir() and (path / "eval_data_manifest.json").is_file()
    )
    if not tasks:
        raise FileNotFoundError(
            f"No eval_data_manifest.json files found under {manifest_root}. Run scripts/prepare_realworld_eval_data.py first."
        )
    return tasks


def selected_tasks(args: argparse.Namespace) -> list[str]:
    tasks = available_tasks(args.manifest_root)
    if args.tasks == ["all"]:
        return tasks
    missing = [task for task in args.tasks if task not in tasks]
    if missing:
        raise ValueError(f"Unknown tasks: {missing}. Available tasks: {tasks}")
    return args.tasks


def load_manifest(manifest_root: Path, task: str) -> dict:
    manifest_path = manifest_root / task / "eval_data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest {manifest_path}")
    return read_json(manifest_path)


def resolve_task_path(task_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        if path.exists():
            return path
        fallback = task_dir / path.name
        if fallback.exists():
            return fallback
        return path
    return task_dir / path


def build_jobs(args: argparse.Namespace) -> tuple[list[Job], list[dict]]:
    requested_human_counts = sorted({int(count) for count in args.human_counts})
    requested_robot_counts = sorted({int(count) for count in args.robot_counts})
    jobs = []
    skipped = []

    for task in selected_tasks(args):
        manifest = load_manifest(args.manifest_root, task)
        task_dir = args.manifest_root / task
        grid = {
            (int(entry["human_demos"]), int(entry["robot_demos"])): entry
            for entry in manifest.get("training_grid", [])
        }

        human_zarr = resolve_task_path(task_dir, manifest["human_zarr"])
        robot_zarr = resolve_task_path(task_dir, manifest["robot_zarr"])

        for human_count in requested_human_counts:
            for robot_count in requested_robot_counts:
                grid_entry = grid.get((human_count, robot_count))
                if grid_entry is None:
                    skipped.append(
                        {
                            "task": task,
                            "human_demos": human_count,
                            "robot_demos": robot_count,
                            "reason": "split unavailable in manifest",
                        }
                    )
                    continue

                human_mask = resolve_task_path(task_dir, grid_entry["human_mask"])
                robot_mask = resolve_task_path(task_dir, grid_entry["robot_mask"])

                missing_inputs = []
                for path in [human_mask, robot_mask]:
                    if not path.is_file():
                        missing_inputs.append(str(path))

                if not args.dry_run:
                    for path in [human_zarr, robot_zarr]:
                        if not path.exists():
                            missing_inputs.append(str(path))

                if missing_inputs:
                    skipped.append(
                        {
                            "task": task,
                            "human_demos": human_count,
                            "robot_demos": robot_count,
                            "reason": "missing inputs",
                            "paths": missing_inputs,
                        }
                    )
                    continue

                jobs.append(
                    Job(
                        task=task,
                        human_demos=human_count,
                        robot_demos=robot_count,
                        human_mask=human_mask,
                        robot_mask=robot_mask,
                        human_zarr=human_zarr,
                        robot_zarr=robot_zarr,
                    )
                )

    return jobs, skipped


def validate_args(args: argparse.Namespace) -> None:
    if not args.devices:
        raise ValueError("--devices must contain at least one device")
    if args.max_concurrent is None:
        args.max_concurrent = len(args.devices)
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")


def build_device_slots(devices: list[str], max_concurrent: int) -> list[str]:
    if max_concurrent <= len(devices):
        return list(devices[:max_concurrent])
    return [devices[index % len(devices)] for index in range(max_concurrent)]


def runtime_device_config(device: str) -> dict:
    if device == "cpu":
        return {
            "visible_device": None,
            "skill_accelerator": "cpu",
            "skill_devices": "1",
            "runtime_device": "cpu",
        }

    if device.startswith("cuda:"):
        visible_device = device.split(":", maxsplit=1)[1]
        return {
            "visible_device": visible_device,
            "skill_accelerator": "gpu",
            "skill_devices": "[0]",
            "runtime_device": "cuda:0",
        }

    raise ValueError(f"Unsupported device '{device}'. Use cpu or cuda:N")


def child_directories(path: Path) -> set[Path]:
    if not path.is_dir():
        return set()
    return {child for child in path.iterdir() if child.is_dir()}


def detect_new_directory(parent: Path, before: set[Path]) -> Path | None:
    after = child_directories(parent)
    new_directories = sorted(after - before, key=lambda entry: entry.stat().st_mtime)
    if new_directories:
        return new_directories[-1]
    if after:
        return sorted(after, key=lambda entry: entry.stat().st_mtime)[-1]
    return None


def job_paths(output_root: Path, job: Job) -> dict:
    job_root = output_root / job.task / f"human_{job.human_demos}_robot_{job.robot_demos}"
    return {
        "job_root": job_root,
        "logs_dir": job_root / "logs",
        "summary_path": job_root / "job_summary.json",
        "skill_dir": job_root / "skill_discovery",
        "bc_root": job_root / "bc",
    }


def skill_checkpoint_candidates(skill_dir: Path, ckpt: int) -> list[Path]:
    return [
        skill_dir / f"epoch={ckpt}.ckpt",
        skill_dir / f"epoch={ckpt:02d}.ckpt",
    ]


def find_skill_checkpoint(skill_dir: Path, ckpt: int) -> Path | None:
    for candidate in skill_checkpoint_candidates(skill_dir, ckpt):
        if candidate.is_file():
            return candidate
    return None


def replay_buffer_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        replay_buffer = ReplayBuffer.create_from_path(str(path), mode="r")
    except Exception:
        return False
    return replay_buffer.n_episodes > 0


def label_outputs_ready(human_proto: Path, robot_proto: Path) -> bool:
    return replay_buffer_ready(human_proto) and replay_buffer_ready(robot_proto)


def build_commands(args: argparse.Namespace, job: Job, device: str, paths: dict) -> dict:
    device_cfg = runtime_device_config(device)
    skill_dir = paths["skill_dir"]
    bc_root = paths["bc_root"]
    human_proto = skill_dir / "human_encode_protos" / f"ckpt_{args.skill_ckpt}" / "human.zarr"
    robot_proto = skill_dir / "encode_protos" / f"ckpt_{args.skill_ckpt}" / "robot.zarr"
    project_name = f"realworld_{job.task}_h{job.human_demos}_r{job.robot_demos}_diffusion_bc"

    skill_overrides = [
        f"hydra.run.dir={hydra_scalar(skill_dir)}",
        f"Trainer.accelerator={device_cfg['skill_accelerator']}",
        f"Trainer.devices={device_cfg['skill_devices']}",
        f"robot_dataset.dataset_paths={hydra_list([job.robot_zarr])}",
        f"robot_dataset.mask={hydra_scalar(job.robot_mask)}",
        f"human_dataset.dataset_paths={hydra_list([job.human_zarr])}",
        f"human_dataset.mask={hydra_scalar(job.human_mask)}",
    ]
    if args.skill_max_epochs is not None:
        skill_overrides.append(f"Trainer.max_epochs={args.skill_max_epochs}")
    if args.skill_save_every is not None:
        skill_overrides.append(f"callback.every_n_epoch={args.skill_save_every}")

    label_overrides = [
        f"exp_path={hydra_scalar(skill_dir)}",
        f"ckpt={args.skill_ckpt}",
        f"device={hydra_scalar(device_cfg['runtime_device'])}",
        f"datasets.human.data_path={hydra_scalar(job.human_zarr)}",
        f"datasets.human.save_path={hydra_scalar(human_proto)}",
        f"datasets.human.mask_path={hydra_scalar(job.human_mask)}",
        f"datasets.robot.data_path={hydra_scalar(job.robot_zarr)}",
        f"datasets.robot.save_path={hydra_scalar(robot_proto)}",
        f"datasets.robot.mask_path={hydra_scalar(job.robot_mask)}",
    ]

    bc_overrides = [
        f"save_dir={hydra_scalar(bc_root)}",
        f"project_name={hydra_scalar(project_name)}",
        f"device={hydra_scalar(device_cfg['runtime_device'])}",
        f"pretrain_path={hydra_scalar(skill_dir)}",
        f"pretrain_ckpt={args.skill_ckpt}",
        f"dataset.data_dirs={hydra_list([job.robot_zarr])}",
        f"dataset.proto_dirs={hydra_list([robot_proto])}",
        f"dataset.mask={hydra_list([job.robot_mask])}",
    ]
    if args.bc_num_epochs is not None:
        bc_overrides.append(f"num_epochs={args.bc_num_epochs}")
    if args.bc_ckpt_frequency is not None:
        bc_overrides.append(f"ckpt_frequency={args.bc_ckpt_frequency}")

    return {
        "skill": [args.python, "scripts/skill_discovery.py", *skill_overrides],
        "label": [args.python, "scripts/label_real_kitchen_dataset.py", *label_overrides],
        "bc": [args.python, "scripts/skill_transfer_composing.py", *bc_overrides],
        "robot_proto": robot_proto,
        "human_proto": human_proto,
    }


def run_stage(command: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write(shlex.join(command) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=repo_root(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise StageFailure(log_path.stem, log_path, result.returncode)


def log(message: str, print_lock: threading.Lock) -> None:
    with print_lock:
        print(message, flush=True)


def run_job(args: argparse.Namespace, job: Job, device: str, semaphore: threading.Semaphore, print_lock: threading.Lock) -> dict:
    with semaphore:
        paths = job_paths(args.output_root, job)
        paths["job_root"].mkdir(parents=True, exist_ok=True)
        paths["logs_dir"].mkdir(parents=True, exist_ok=True)

        existing_summary = None
        if paths["summary_path"].is_file():
            try:
                existing_summary = read_json(paths["summary_path"])
            except json.JSONDecodeError:
                existing_summary = None
        if args.skip_existing and existing_summary and existing_summary.get("status") == "succeeded":
            log(f"[skip] {job.name}", print_lock)
            return existing_summary

        commands = build_commands(args, job, device, paths)
        existing_skill_checkpoint = find_skill_checkpoint(paths["skill_dir"], args.skill_ckpt)
        skip_skill = existing_skill_checkpoint is not None
        skip_label = skip_skill and label_outputs_ready(commands["human_proto"], commands["robot_proto"])
        summary = {
            "task": job.task,
            "human_demos": job.human_demos,
            "robot_demos": job.robot_demos,
            "device_slot": device,
            "job_root": str(paths["job_root"]),
            "skill_dir": str(paths["skill_dir"]),
            "robot_proto": str(commands["robot_proto"]),
            "human_proto": str(commands["human_proto"]),
            "skill_checkpoint": str(existing_skill_checkpoint)
            if existing_skill_checkpoint is not None
            else str(skill_checkpoint_candidates(paths["skill_dir"], args.skill_ckpt)[0]),
            "status": "running",
            "started_at": timestamp(),
            "commands": {
                stage: shlex.join(command)
                for stage, command in commands.items()
                if isinstance(command, list)
            },
            "inputs": asdict(job),
        }
        write_json(paths["summary_path"], summary)

        if args.dry_run:
            summary["status"] = "dry_run"
            summary["finished_at"] = timestamp()
            write_json(paths["summary_path"], summary)
            log(f"[dry-run] {job.name}", print_lock)
            return summary

        env = os.environ.copy()
        visible_device = runtime_device_config(device)["visible_device"]
        if visible_device is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = visible_device
        if args.wandb_entity:
            env["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_mode:
            env["WANDB_MODE"] = args.wandb_mode

        try:
            log(f"[start] {job.name} on {device}", print_lock)
            if skip_skill:
                summary["skill_status"] = "skipped_existing"
                summary["skill_finished_at"] = (
                    existing_summary.get("skill_finished_at")
                    if existing_summary and existing_summary.get("skill_finished_at")
                    else timestamp()
                )
                write_json(paths["summary_path"], summary)
                log(f"[skip-skill] {job.name} using {existing_skill_checkpoint.name}", print_lock)
            else:
                run_stage(commands["skill"], paths["logs_dir"] / "skill.log", env)
                summary["skill_status"] = "succeeded"
                summary["skill_finished_at"] = timestamp()
                write_json(paths["summary_path"], summary)

            if skip_label:
                summary["label_status"] = "skipped_existing"
                summary["label_finished_at"] = (
                    existing_summary.get("label_finished_at")
                    if existing_summary and existing_summary.get("label_finished_at")
                    else timestamp()
                )
                write_json(paths["summary_path"], summary)
                log(f"[skip-label] {job.name}", print_lock)
            else:
                run_stage(commands["label"], paths["logs_dir"] / "label.log", env)
                summary["label_status"] = "succeeded"
                summary["label_finished_at"] = timestamp()
                write_json(paths["summary_path"], summary)

            bc_before = child_directories(paths["bc_root"])
            run_stage(commands["bc"], paths["logs_dir"] / "bc.log", env)
            bc_run_dir = detect_new_directory(paths["bc_root"], bc_before)
            summary["bc_run_dir"] = str(bc_run_dir) if bc_run_dir is not None else None
            summary["bc_status"] = "succeeded"
            summary["status"] = "succeeded"
            summary["finished_at"] = timestamp()
            write_json(paths["summary_path"], summary)
            log(f"[done] {job.name}", print_lock)
            return summary
        except StageFailure as error:
            summary["status"] = "failed"
            summary[f"{error.stage}_status"] = "failed"
            summary["failed_stage"] = error.stage
            summary["returncode"] = error.returncode
            summary["failed_log"] = str(error.log_path)
            summary["finished_at"] = timestamp()
            write_json(paths["summary_path"], summary)
            log(f"[failed] {job.name} stage={error.stage} log={error.log_path}", print_lock)
            return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    jobs, skipped = build_jobs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not jobs:
        payload = {
            "status": "no_jobs",
            "skipped": skipped,
            "generated_at": timestamp(),
        }
        write_json(args.output_root / "run_summary.json", payload)
        print("No runnable jobs found.", flush=True)
        return 1

    print(f"Planned jobs: {len(jobs)}", flush=True)
    if skipped:
        print(f"Skipped jobs: {len(skipped)}", flush=True)

    semaphore = threading.Semaphore(args.max_concurrent)
    print_lock = threading.Lock()
    device_pool = queue.Queue()
    device_slots = build_device_slots(args.devices, args.max_concurrent)
    for device in device_slots:
        device_pool.put(device)

    results = []
    with ThreadPoolExecutor(max_workers=min(len(jobs), len(device_slots))) as executor:
        future_to_job = {}
        for job in jobs:
            device = device_pool.get()

            def task_wrapper(current_job: Job, current_device: str):
                try:
                    return run_job(args, current_job, current_device, semaphore, print_lock)
                finally:
                    device_pool.put(current_device)

            future = executor.submit(task_wrapper, job, device)
            future_to_job[future] = job

        for future in as_completed(future_to_job):
            results.append(future.result())

    succeeded = [result for result in results if result.get("status") == "succeeded"]
    failed = [result for result in results if result.get("status") == "failed"]
    dry_runs = [result for result in results if result.get("status") == "dry_run"]
    payload = {
        "generated_at": timestamp(),
        "jobs": len(jobs),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "dry_run": len(dry_runs),
        "skipped": skipped,
        "results": results,
    }
    write_json(args.output_root / "run_summary.json", payload)

    if failed:
        print(f"Failed jobs: {len(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())