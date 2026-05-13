from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Write absolute task paths rooted here. If omitted, manifests are rewritten to task-relative paths.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as file:
        json.dump(payload, file, indent=2)
    temp_path.replace(path)


def available_tasks(manifest_root: Path) -> list[str]:
    if not manifest_root.is_dir():
        raise FileNotFoundError(f"Missing manifest root {manifest_root}")
    tasks = sorted(
        path.name
        for path in manifest_root.iterdir()
        if path.is_dir() and (path / "eval_data_manifest.json").is_file()
    )
    if not tasks:
        raise FileNotFoundError(f"No eval_data_manifest.json files found under {manifest_root}")
    return tasks


def selected_tasks(args: argparse.Namespace) -> list[str]:
    tasks = available_tasks(args.manifest_root)
    if args.tasks == ["all"]:
        return tasks
    missing = [task for task in args.tasks if task not in tasks]
    if missing:
        raise ValueError(f"Unknown tasks: {missing}. Available tasks: {tasks}")
    return args.tasks


def format_task_path(task_name: str, relative_name: str, target_root: Path | None) -> str:
    if target_root is None:
        return relative_name
    return str(target_root / task_name / relative_name)


def resolve_manifest_path(task_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return task_dir / path


def expected_paths(task_name: str, manifest: dict, target_root: Path | None) -> dict:
    human_masks = {
        str(count): format_task_path(task_name, f"human_mask_{int(count)}.json", target_root)
        for count in manifest.get("human_masks", {}).keys()
    }
    robot_masks = {
        str(count): format_task_path(task_name, f"robot_mask_{int(count)}.json", target_root)
        for count in manifest.get("robot_masks", {}).keys()
    }
    training_grid = []
    for entry in manifest.get("training_grid", []):
        updated_entry = dict(entry)
        updated_entry["human_mask"] = format_task_path(
            task_name,
            f"human_mask_{int(entry['human_demos'])}.json",
            target_root,
        )
        updated_entry["robot_mask"] = format_task_path(
            task_name,
            f"robot_mask_{int(entry['robot_demos'])}.json",
            target_root,
        )
        training_grid.append(updated_entry)
    return {
        "human_zarr": format_task_path(task_name, "human.zarr", target_root),
        "robot_zarr": format_task_path(task_name, "robot.zarr", target_root),
        "human_masks": human_masks,
        "robot_masks": robot_masks,
        "training_grid": training_grid,
    }


def validate_manifest(task_dir: Path, manifest: dict) -> list[str]:
    missing = []
    path_items = [
        ("human_zarr", manifest["human_zarr"]),
        ("robot_zarr", manifest["robot_zarr"]),
    ]
    path_items.extend((f"human_masks[{count}]", path) for count, path in manifest.get("human_masks", {}).items())
    path_items.extend((f"robot_masks[{count}]", path) for count, path in manifest.get("robot_masks", {}).items())

    for label, raw_path in path_items:
        resolved = resolve_manifest_path(task_dir, raw_path)
        exists = resolved.is_file() if resolved.suffix == ".json" else resolved.exists()
        if not exists:
            missing.append(f"{label}: {resolved}")

    for index, entry in enumerate(manifest.get("training_grid", [])):
        for key in ["human_mask", "robot_mask"]:
            resolved = resolve_manifest_path(task_dir, entry[key])
            if not resolved.is_file():
                missing.append(f"training_grid[{index}].{key}: {resolved}")

    return missing


def rewrite_task(manifest_root: Path, task_name: str, target_root: Path | None, dry_run: bool) -> None:
    task_dir = manifest_root / task_name
    manifest_path = task_dir / "eval_data_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(expected_paths(task_name, manifest, target_root))
    missing = validate_manifest(task_dir, manifest)
    if missing:
        raise FileNotFoundError(
            "Manifest rewrite validation failed for "
            f"{manifest_path}:\n" + "\n".join(f"- {item}" for item in missing)
        )
    if dry_run:
        print(f"[dry-run] would rewrite {manifest_path}")
        return
    write_json(manifest_path, manifest)
    print(f"rewrote {manifest_path}")


def main() -> None:
    args = parse_args()
    for task_name in selected_tasks(args):
        rewrite_task(args.manifest_root, task_name, args.target_root, args.dry_run)


if __name__ == "__main__":
    main()