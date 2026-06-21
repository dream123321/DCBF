import numpy as np

from .mlp_encoding_extract import decode


def _safe_positive_bins(number_of_bins):
    return max(1, int(number_of_bins))


def Freedman_Diaconis_bins(data, dq_width_factor=1.0):
    data = np.asarray(data)
    if data.size <= 1:
        return 1

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    n = len(data)

    if iqr > 0:
        bin_width = 2 * iqr / (n ** (1 / 3)) * dq_width_factor
    else:
        std_dev = np.std(data)
        if std_dev > 0:
            bin_width = 3.5 * std_dev / (n ** (1 / 3)) * dq_width_factor
        else:
            return 1

    data_range = float(np.ptp(data))
    if data_range == 0 or bin_width <= 0:
        return 1
    return _safe_positive_bins(np.ceil(data_range / bin_width))


def scott(data, dq_width_factor=1.0):
    data = np.asarray(data)
    if data.size <= 1:
        return 1
    std_dev = np.std(data)
    if std_dev == 0:
        return 1
    n = len(data)
    bin_width_scott = 3.5 * std_dev / (n ** (1 / 3)) * dq_width_factor
    data_range = float(np.ptp(data))
    if data_range == 0 or bin_width_scott <= 0:
        return 1
    return _safe_positive_bins(np.ceil(data_range / bin_width_scott))


def distribution(array_data, dq_width, dq_width_method, fig_title, body, plot_model, dq_width_factor=1.0, state_population=0):
    state_population = max(0, int(state_population))
    D = len(array_data[0])
    zero_freq_intervals_list = []
    max_min = []
    bins = []
    array_data = np.array(array_data)
    axes = None
    if plot_model:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(D, 1, figsize=(8, D * 4))
        axes = np.atleast_1d(axes)
        fig.suptitle(f"body_{body} distribution of type_{fig_title} (method_{dq_width_method})")

    for i in range(D):
        new_data = array_data[:, i]
        max_min.append([max(new_data), min(new_data)])
        if dq_width_method == "Freedman_Diaconis":
            bin_count = Freedman_Diaconis_bins(new_data, dq_width_factor=dq_width_factor)
        elif dq_width_method == "self_input":
            denom = float(dq_width)
            if denom <= 0:
                raise ValueError("dq_width must be > 0 when dq_width_method is self_input")
            bin_count = _safe_positive_bins(np.ceil((max(new_data) - min(new_data)) / denom))
        elif dq_width_method == "scott":
            bin_count = scott(new_data, dq_width_factor=dq_width_factor)
        elif dq_width_method == "std":
            std_dev = float(np.std(new_data))
            if std_dev == 0:
                bin_count = 1
            else:
                bin_count = _safe_positive_bins(np.ceil((max(new_data) - min(new_data)) / (std_dev / 10)))
        else:
            raise ValueError("dq_width_method does not exist!")

        bins.append(bin_count)
        if plot_model:
            frequencies, bin_edges, _ = axes[i].hist(new_data, bins=bin_count, alpha=0.6, color="g", edgecolor="black")
        else:
            frequencies, bin_edges = np.histogram(new_data, bins=bin_count)
        zero_freq_intervals = [
            [bin_edges[index], bin_edges[index + 1]]
            for index in range(len(bin_edges) - 1)
            if frequencies[index] <= state_population
        ]
        zero_freq_intervals_list.append(zero_freq_intervals)
        if plot_model:
            axes[i].set_title(f"Dimension {i} | low-count threshold <= {state_population}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel(f"Frequency (<= {state_population} treated as uncovered)")

    if plot_model:
        plt.tight_layout()
        plt.savefig(f"body_{body} distribution of type_{fig_title}", dpi=300)
        plt.close()
    return zero_freq_intervals_list, max_min, bins


def worker(args):
    type_atoms, dq_width, dq_width_method, fig_title, body, plot_model, dq_width_factor, state_population = args
    stru_temp = [atom[:-1] for atom in type_atoms]
    tt = np.array(stru_temp)
    zero_freq_intervals_list, max_min, bins = distribution(
        tt,
        dq_width,
        dq_width_method,
        fig_title,
        body,
        plot_model,
        dq_width_factor=dq_width_factor,
        state_population=state_population,
    )
    return zero_freq_intervals_list, max_min, bins


def data_base_distribution(data_base_data, dq_width, dq_width_method, body, plot_model, dq_width_factor=1.0, state_population=0):
    train_data = decode(data_base_data)
    state_population = max(0, int(state_population))
    large_zero_freq_intervals_list = []
    large_max_min = []
    large_bins = []

    for type_index, type_atoms in enumerate(train_data):
        if type_atoms:
            zero_freq_intervals_list, max_min, bins = worker(
                (type_atoms, dq_width, dq_width_method, str(type_index), body, plot_model, dq_width_factor, state_population)
            )
        else:
            zero_freq_intervals_list, max_min, bins = [], [], []
        large_zero_freq_intervals_list.append(zero_freq_intervals_list)
        large_max_min.append(max_min)
        large_bins.append(bins)

    return large_zero_freq_intervals_list, large_max_min, large_bins
