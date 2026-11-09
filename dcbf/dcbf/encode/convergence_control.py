from __future__ import annotations

import json
from pathlib import Path


def coverage_target_from_schedule(coverage_threshold_schedule):
    values = []
    for stage in coverage_threshold_schedule:
        if isinstance(stage, (int, float)):
            values.append(float(stage))
        else:
            values.extend(float(value) for value in stage)
    if not values:
        return 100.0
    return max(values)


def convergence_history_path(pwd):
    return Path(pwd).resolve().parent / "coverage_convergence_history.json"


def load_convergence_history(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_convergence_history(path, history):
    path = Path(path)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def update_metric_history(path, metric_key, gen_index, metric_value):
    history = load_convergence_history(path)
    metric_history = list(history.get(metric_key, []))
    metric_history = [item for item in metric_history if int(item.get("gen", -1)) != int(gen_index)]
    metric_history.append({"gen": int(gen_index), "metric": float(metric_value)})
    metric_history.sort(key=lambda item: int(item["gen"]))
    history[metric_key] = metric_history
    save_convergence_history(path, history)
    return metric_history


def evaluate_metric_convergence(metric_history, target, plateau_generations=None, min_coverage_delta=None):
    current_metric = float(metric_history[-1]["metric"]) if metric_history else 100.0
    hard_converged = current_metric >= float(target)

    plateau_converged = False
    recent_improvements = []
    if (
        plateau_generations is not None
        and min_coverage_delta is not None
        and int(plateau_generations) >= 2
        and len(metric_history) >= int(plateau_generations)
        and not hard_converged
    ):
        recent = metric_history[-int(plateau_generations):]
        recent_improvements = [
            float(recent[index]["metric"]) - float(recent[index - 1]["metric"])
            for index in range(1, len(recent))
        ]
        plateau_converged = all(improvement < float(min_coverage_delta) for improvement in recent_improvements)

    return {
        "metric": current_metric,
        "target": float(target),
        "hard_converged": hard_converged,
        "plateau_converged": plateau_converged,
        "converged": hard_converged or plateau_converged,
        "recent_improvements": recent_improvements,
    }


def log_plateau_convergence(
    logger,
    encoding_result,
    mean_result,
    mean_descriptor_enabled,
    plateau_generations,
    min_coverage_delta,
):
    results = [("encoding_body_coverage", encoding_result)]
    if mean_descriptor_enabled:
        results.append(("mean_descriptor", mean_result))

    if not any(result.get("plateau_converged", False) for _, result in results):
        return

    statuses = []
    for metric_name, result in results:
        if result.get("plateau_converged", False):
            status = "plateau"
        elif result.get("hard_converged", False):
            status = "hard_threshold"
        else:
            status = "not_converged"
        statuses.append(f"{metric_name}={status}")
        logger.info(
            "Coverage convergence | metric=%s | reason=%s | current=%.4f%% | "
            "hard_target=%.4f%% | recent_deltas=%s | plateau_generations=%s | "
            "min_coverage_delta=%.4f",
            metric_name,
            status,
            float(result.get("metric", 100.0)),
            float(result.get("target", 100.0)),
            [round(float(value), 4) for value in result.get("recent_improvements", [])],
            plateau_generations,
            float(min_coverage_delta),
        )
    logger.info("Active-learning loop ended | reason=coverage_plateau | %s", " | ".join(statuses))
