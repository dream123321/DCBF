import os
from collections import Counter

import numpy as np

from .mlp_encoding_extract import decode
from .coverage_policy import ensure_decoded
from .selection_core import label_covered_values
from .descriptor_store import column, numeric_data, values_and_indices


def md_sub_extract(array_data, type_zero_freq_intervals_list, max_min, coverage_rate_method, labels=True):
    coverage_rate_list = []

    array_data = numeric_data(array_data)
    D = array_data.shape[1]
    new_lable_array = np.zeros(len(array_data), dtype=int) if labels else None
    for i in range(D):
        new_data = column(array_data, i)
        label_array = label_covered_values(new_data, type_zero_freq_intervals_list[i], max_min[i])
        if labels:
            new_lable_array += label_array.astype(int)
        coverage_rate_list.append(float(np.mean(label_array) * 100))

    if coverage_rate_method == "mean":
        coverage_rate = sum(coverage_rate_list) / len(coverage_rate_list)
    elif coverage_rate_method == "min":
        coverage_rate = min(coverage_rate_list)
    else:
        raise ValueError("coverage_rate_method has only mean and min!")
    return new_lable_array, coverage_rate


def coverage_rate(
    data,
    large_zero_freq_intervals_list,
    large_max_min,
    body,
    plot_out,
    coverage_rate_method,
    plot_model,
    plot_suffix="",
):
    md_data = ensure_decoded(data)
    type_coverage_rate = []
    type_coverage_rate_100 = []
    type_coverage_rate_index = []

    for type_index, (type_atoms, type_zero_freq_intervals_list, max_min) in enumerate(
        zip(md_data, large_zero_freq_intervals_list, large_max_min)
    ):
        tt, _ = values_and_indices(type_atoms)
        type_coverage_rate_100.append(100)
        if not len(tt):
            continue
        type_coverage_rate_index.append(type_index)
        lable_array, single_coverage_rate = md_sub_extract(
            tt, type_zero_freq_intervals_list, max_min, coverage_rate_method, labels=plot_model)
        if plot_model:
            import matplotlib.pyplot as plt

            D = tt.shape[1]
            plot_data = [column(tt, i) for i in range(D)]

            cmap = plt.cm.get_cmap("viridis", D + 1)
            scatter = plt.scatter(plot_data[0], plot_data[1], s=0.2, c=lable_array, cmap=cmap, vmin=-0.5, vmax=D + 0.5)
            element_counts = Counter(lable_array)
            sorted_counts = sorted(element_counts.items(), key=lambda item: item[0])
            count_str = "\n".join(f"{elem}: {count}" for elem, count in sorted_counts)

            cbar = plt.colorbar(scatter, ticks=np.arange(0, D + 1, 1))
            cbar.ax.set_yticklabels(np.arange(0, D + 1, 1))

            plt.title(f"{body}_body_type_{str(type_index)} mean_coverage_rate:{round(single_coverage_rate, 5)}%")
            plt.xlabel("Dimension_0")
            plt.ylabel("Dimension_1")
            plt.text(0.95, 0.95, count_str, transform=plt.gca().transAxes, verticalalignment="top", horizontalalignment="right")
            plt.savefig(os.path.join(plot_out, f"{body}_body_type_{str(type_index)}{plot_suffix}.png"), dpi=300)
            plt.close()
        type_coverage_rate.append(single_coverage_rate)

    for index, value in zip(type_coverage_rate_index, type_coverage_rate):
        type_coverage_rate_100[index] = value
    return type_coverage_rate_100
