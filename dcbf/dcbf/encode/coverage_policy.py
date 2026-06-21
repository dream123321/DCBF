from __future__ import annotations

from collections import OrderedDict
from numbers import Real
import os

from .mlp_encoding_extract import decode


def ensure_decoded(data_or_path):
    if isinstance(data_or_path, (str, os.PathLike)):
        return decode(str(data_or_path))
    return data_or_path


def _as_float_list(values):
    return [float(value) for value in values]


def normalize_coverage_thresholds(coverage_rate_threshold, element_count):
    normalized = []
    scalar_mode = True
    for stage in coverage_rate_threshold:
        if isinstance(stage, Real):
            row = [float(stage)] * element_count
        else:
            row = _as_float_list(stage)
            if len(row) == 1:
                row = row * element_count
            elif len(row) != element_count:
                raise ValueError(
                    f"coverage_rate_threshold inner length must be 1 or {element_count}, got {len(row)}"
                )
            scalar_mode = False
        normalized.append(row)
    return normalized, scalar_mode


def determine_structure_budget(element_coverages, stru_num, coverage_rate_threshold):
    normalized_thresholds, scalar_mode = normalize_coverage_thresholds(coverage_rate_threshold, len(element_coverages))

    stage_pairs = list(zip(normalized_thresholds, stru_num))
    if scalar_mode:
        stage_pairs.sort(key=lambda item: item[0][0])

    max_thresholds = [max(stage[element_index] for stage in normalized_thresholds) for element_index in range(len(element_coverages))]
    convergence = all(coverage > threshold for coverage, threshold in zip(element_coverages, max_thresholds))

    real_stru_num = 0
    for thresholds, budget in stage_pairs:
        if any(coverage < threshold for coverage, threshold in zip(element_coverages, thresholds)):
            real_stru_num = budget
            break

    return real_stru_num, convergence, normalized_thresholds


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
    index_set = set(structure_indices)
    filtered = []
    for type_atoms in ensure_decoded(decoded_data):
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
