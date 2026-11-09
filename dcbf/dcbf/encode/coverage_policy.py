from __future__ import annotations

from collections import OrderedDict
import math
from numbers import Real
import os

from .mlp_encoding_extract import decode


SELECTION_BUDGET_SCOPES = {"per_configuration", "all_configurations"}


def normalize_selection_budget_scope(value):
    scope = "per_configuration" if value is None else str(value).strip().lower()
    if scope not in SELECTION_BUDGET_SCOPES:
        raise ValueError(
            "selection_budget_scope must be 'per_configuration' or "
            "'all_configurations'"
        )
    return scope


def validate_budget_schedule(stru_num):
    """Validate only the per-stage structure budgets.

    Reduce treats ``selection_budget_by_coverage`` as optional: a one-shot
    selection runs without any budget truncation.  Keeping the budget and
    threshold validation separate lets either field be omitted while the
    update/SCF loop still requires both.
    """
    try:
        budgets = list(stru_num)
    except TypeError as exc:
        raise ValueError("selection_budget_schedule must be a list") from exc
    if not budgets:
        raise ValueError("selection_budget_schedule must be non-empty")
    normalized_budgets = []
    for value in budgets:
        if isinstance(value, bool):
            raise ValueError("selection_budget_schedule entries must be non-negative integers")
        try:
            budget = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "selection_budget_schedule entries must be non-negative integers"
            ) from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("selection_budget_schedule entries must be non-negative integers")
        if budget < 0:
            raise ValueError("selection_budget_schedule entries must be non-negative integers")
        normalized_budgets.append(budget)
    return normalized_budgets


def validate_coverage_threshold_schedule(coverage_rate_threshold, element_count):
    """Validate only the per-stage coverage targets."""
    if element_count < 1:
        raise ValueError("selection schedules require at least one element")
    try:
        thresholds = list(coverage_rate_threshold)
    except TypeError as exc:
        raise ValueError("coverage_threshold_schedule must be a list") from exc
    if not thresholds:
        raise ValueError("coverage_threshold_schedule must be non-empty")
    normalized_thresholds, _ = normalize_coverage_thresholds(thresholds, element_count)
    for row in normalized_thresholds:
        if any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in row):
            raise ValueError(
                "coverage_threshold_schedule entries must be finite numbers between 0 and 100"
            )
    for element_index in range(element_count):
        values = [row[element_index] for row in normalized_thresholds]
        if any(current < previous for previous, current in zip(values, values[1:])):
            raise ValueError("coverage_threshold_schedule must be non-decreasing for every element")
    return normalized_thresholds


def validate_selection_schedules(stru_num, coverage_rate_threshold, element_count):
    if element_count < 1:
        raise ValueError("selection schedules require at least one element")
    try:
        budgets = list(stru_num)
        thresholds = list(coverage_rate_threshold)
    except TypeError as exc:
        raise ValueError(
            "selection_budget_schedule and coverage_threshold_schedule must be lists"
        ) from exc
    if not budgets or len(budgets) != len(thresholds):
        raise ValueError(
            "selection_budget_schedule and coverage_threshold_schedule must be "
            "non-empty and have the same length"
        )
    normalized_budgets = validate_budget_schedule(budgets)
    normalized_thresholds = validate_coverage_threshold_schedule(thresholds, element_count)
    return normalized_budgets, normalized_thresholds


def stable_unique(indices):
    seen = set()
    output = []
    for raw_index in indices:
        index = int(raw_index)
        if index not in seen:
            seen.add(index)
            output.append(index)
    return output


def strict_budget_selection(indices, budget):
    return stable_unique(indices)[: max(0, int(budget))]


def ensure_decoded(data_or_path):
    if isinstance(data_or_path, (str, os.PathLike)):
        return decode(str(data_or_path))
    return data_or_path


def _as_float_list(values):
    output = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("coverage_threshold_schedule entries must be finite numbers")
        output.append(float(value))
    return output


def normalize_coverage_thresholds(coverage_rate_threshold, element_count):
    normalized = []
    scalar_mode = True
    for stage in coverage_rate_threshold:
        if isinstance(stage, bool):
            raise ValueError("coverage_threshold_schedule entries must be finite numbers")
        if isinstance(stage, Real):
            row = [float(stage)] * element_count
        else:
            row = _as_float_list(stage)
            if len(row) != element_count:
                raise ValueError(
                    "coverage_threshold_schedule element-resolved row length must "
                    f"equal the element count {element_count}, got {len(row)}"
                )
            scalar_mode = False
        normalized.append(row)
    return normalized, scalar_mode


def determine_structure_budget_with_stage(
    element_coverages,
    stru_num,
    coverage_rate_threshold,
):
    """Return the active stage together with this round's budget.

    ``element_coverages`` is the per-element coverage of the current dataset.
    ``convergence`` becomes True once every element has reached the final
    (largest) stage threshold, which is the Reduce stop condition.  Otherwise
    the budget belongs to the first stage that is still below its target, so
    the multi-round loop advances stages from coverage alone instead of from a
    round counter.

    Returns ``(budget, convergence, normalized_thresholds, stage_index,
    stage_target)``.  ``stage_index`` is None and ``stage_target`` is the final
    thresholds when the run has converged.
    """
    stru_num, normalized_thresholds = validate_selection_schedules(
        stru_num,
        coverage_rate_threshold,
        len(element_coverages),
    )
    max_thresholds = [
        max(stage[element_index] for stage in normalized_thresholds)
        for element_index in range(len(element_coverages))
    ]
    convergence = all(
        coverage >= threshold for coverage, threshold in zip(element_coverages, max_thresholds)
    )
    if convergence:
        return 0, True, normalized_thresholds, None, max_thresholds

    for stage_index, (thresholds, budget) in enumerate(
        zip(normalized_thresholds, stru_num)
    ):
        if any(coverage < threshold for coverage, threshold in zip(element_coverages, thresholds)):
            return int(budget), False, normalized_thresholds, stage_index, list(thresholds)
    # validate_selection_schedules enforces non-decreasing per-element targets,
    # so an unconverged value always matches at least one stage.  Guard anyway
    # instead of silently returning a stale budget.
    raise RuntimeError("Internal error: no coverage stage matched an unconverged coverage")


def determine_structure_budget(element_coverages, stru_num, coverage_rate_threshold):
    budget, convergence, normalized_thresholds, _, _ = determine_structure_budget_with_stage(
        element_coverages,
        stru_num,
        coverage_rate_threshold,
    )
    return budget, convergence, normalized_thresholds


def max_element_thresholds(coverage_rate_threshold, element_count):
    normalized_thresholds, _ = normalize_coverage_thresholds(coverage_rate_threshold, element_count)
    return [max(stage[element_index] for stage in normalized_thresholds) for element_index in range(element_count)]


def aggregate_element_coverages(coverage_batches, element_count):
    element_coverages = [100.0] * element_count
    for batch in coverage_batches:
        for body_rates in batch:
            for element_index, rate in enumerate(body_rates):
                element_coverages[element_index] = min(element_coverages[element_index], float(rate))
    return element_coverages


def select_per_configuration_candidates(
    configuration_groups,
    body_configuration_coverages,
    body_list,
    aee_candidates,
    mean_candidates,
    mean_configuration_coverages,
    mean_descriptor_enabled,
    selection_budget_schedule,
    coverage_threshold_schedule,
    element_count,
):
    selected = []
    budgets = OrderedDict()
    mean_thresholds = scalar_thresholds_for_mean_descriptor(
        coverage_threshold_schedule,
        element_count,
    )
    for config_name, structure_indices in configuration_groups.items():
        body_rates = [
            body_configuration_coverages.get(body, {}).get(config_name)
            for body in body_list
        ]
        body_rates = [rates for rates in body_rates if rates]
        element_coverages = aggregate_element_coverages(
            [[rates] for rates in body_rates],
            element_count,
        )
        budget, _, _ = determine_structure_budget(
            element_coverages,
            selection_budget_schedule,
            coverage_threshold_schedule,
        )
        if mean_descriptor_enabled:
            mean_rates = mean_configuration_coverages.get(config_name) or [100.0]
            mean_budget, _, _ = determine_structure_budget(
                [min(float(rate) for rate in mean_rates)],
                selection_budget_schedule,
                mean_thresholds,
            )
            budget = max(budget, mean_budget)

        config_index_set = set(structure_indices)
        config_mean = [index for index in mean_candidates if index in config_index_set]
        config_aee = list(aee_candidates.get(config_name, []))
        config_selected = strict_budget_selection(
            config_mean + [index for index in config_aee if index not in config_mean],
            budget,
        )
        budgets[config_name] = budget
        selected.extend(config_selected)
    return stable_unique(selected), budgets


def scalar_thresholds_for_mean_descriptor(coverage_rate_threshold, element_count):
    normalized_thresholds, scalar_mode = normalize_coverage_thresholds(coverage_rate_threshold, element_count)
    reduced = [max(stage) for stage in normalized_thresholds]
    if scalar_mode:
        reduced = sorted(reduced)
    return reduced


def build_configuration_groups(dirs, dirs_stru_counts):
    groups = OrderedDict()
    start_index = 0
    for path, structure_count in zip(dirs, dirs_stru_counts):
        config_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        index_group = groups.setdefault(config_name, [])
        index_group.extend(range(start_index, start_index + structure_count))
        start_index += structure_count
    return groups


def slice_decoded_by_indices(decoded_data, structure_indices):
    from .descriptor_store import DescriptorRows
    index_set = set(structure_indices)
    filtered = []
    for type_atoms in ensure_decoded(decoded_data):
        if isinstance(type_atoms, DescriptorRows):
            filtered.append(type_atoms.select_frames(structure_indices))
        else:
            filtered.append([atom for atom in type_atoms if atom[-1] in index_set])
    return filtered


def summarize_configuration_coverages(configuration_coverages, digits=4):
    return {
        config_name: [round(float(rate), digits) for rate in rates]
        for config_name, rates in configuration_coverages.items()
    }


def count_selected_by_configuration(selected_indices, configuration_groups):
    selected_index_set = set(selected_indices)
    return {
        config_name: sum(1 for index in indices if index in selected_index_set)
        for config_name, indices in configuration_groups.items()
    }
