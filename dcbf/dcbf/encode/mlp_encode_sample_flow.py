import os
from ase.io import iread,write
import numpy as np
from .coverage_policy import (
    aggregate_element_coverages,
    build_configuration_groups,
    count_selected_by_configuration,
    determine_structure_budget,
    ensure_decoded,
    max_element_thresholds,
    normalize_selection_budget_scope,
    scalar_thresholds_for_mean_descriptor,
    select_per_configuration_candidates,
    slice_decoded_by_indices,
    summarize_configuration_coverages,
)
from .convergence_control import (
    convergence_history_path,
    coverage_target_from_schedule,
    evaluate_metric_convergence,
    log_plateau_convergence,
    update_metric_history,
)
from .mlp_encoding_extract import des_out2pkl
from .mean_encoding_extract import mean_des_out2pkl
from .descriptor_store import build_descriptor_store, column, numeric_data, values_and_indices
from .compact_indices import group_compact_indices
from .selected_frames import SelectedFrames
from ..memory_guard import descriptor_stage, stage_progress
import time
from .coverage_rate import coverage_rate
from .find_min_cover_set import (
    find_min_cover_set,
    fwss,
    fwss_plus_mean_select_index,
    rank_min_cover_indices,
)
from .dimension_min_cover import (
    build_dimension_tasks,
    merge_dimension_tasks,
    normalize_dimension_min_cover_workers,
    solve_dimension_tasks,
)
import itertools
from .file_conversion import dump2cfg, cfg2xyz, remove
from .mlp_mul_encode import mul_encode
from .cfg_descriptor_encode import encode_cfg_parallel
from .mlp_return_strupkl import mlp_return_strupkl
import glob
from .data_distri import data_base_distribution
from .mean_select import mean_pre_sample_flow
from .selection_core import group_structure_indices_by_interval
from ..npt_volume_filter import (
    build_volume_intake_stats,
    case_dir_context,
    format_volume_summary_message,
    format_volume_truncation_message,
    npt_frame_volume_guard,
    write_npt_volume_filter_report,
)
from ..das.calc_ensemble_ambiguity import check_lmp_error
from ..path_names import MD_WORK_DIR, SUS2_MODEL_DIR
from ..runtime_config import build_scheduler_spec, load_runtime_config


def _rates_reach_thresholds(rates, thresholds):
    if rates is None or len(rates) != len(thresholds):
        return False
    return all(float(rate) >= float(threshold) for rate, threshold in zip(rates, thresholds))


def _next_md_configurations(
    configuration_groups,
    mean_descriptor_enabled,
    mean_configuration_coverages,
    body_configuration_coverages,
    body_list,
    mean_target,
    element_targets,
):
    next_md_names = []
    status = {}
    for config_name in configuration_groups:
        metric_status = {}
        enabled_results = []
        if mean_descriptor_enabled:
            mean_rates = mean_configuration_coverages.get(config_name)
            mean_reached = bool(mean_rates) and all(float(rate) >= float(mean_target) for rate in mean_rates)
            metric_status["mean_descriptor"] = mean_reached
            enabled_results.append(mean_reached)

        for body in body_list:
            body_rates = body_configuration_coverages.get(body, {}).get(config_name)
            body_reached = _rates_reach_thresholds(body_rates, element_targets)
            metric_status[f"{body}_body"] = body_reached
            enabled_results.append(body_reached)

        hard_converged = bool(enabled_results) and all(enabled_results)
        metric_status["hard_converged"] = hard_converged
        metric_status["next_md"] = not hard_converged
        status[config_name] = metric_status
        if not hard_converged:
            next_md_names.append(config_name)
    return next_md_names, status


def _report_volume_truncation(logger, workspace, volume_stats, factor):
    """Report the MD frames that were truncated by the cell-volume guard.

    Returns the structures that lost frames, which have to keep sampling: an expanded
    cell means the trajectory left the region the current MLIP is valid for.
    """
    stats = build_volume_intake_stats(volume_stats)
    write_npt_volume_filter_report(workspace, factor, stats)
    if factor is None:
        return set()

    for case_dir, detail in volume_stats.items():
        if not detail["dropped"]:
            continue
        structure_name, case_name = case_dir_context(case_dir)
        error, message = check_lmp_error(case_dir)
        logger.warning(
            format_volume_truncation_message(
                structure_name,
                case_name,
                detail,
                factor,
                message if error else "",
            )
        )
    if stats["removed_count"]:
        logger.warning(format_volume_summary_message(factor, stats))
    return set(stats["dropped_configurations"])


def delete_md_data(md_path,dirs):
    '''删除md相关文件'''
    files_to_delete = glob.glob(os.path.join(md_path, 'md*'))
    gen_0_files_to_delete = glob.glob(os.path.join(md_path, 'gen*'))
    for file in files_to_delete:
        if os.path.isfile(file) and not file.endswith('.pkl'):
            remove(file)
    for file in gen_0_files_to_delete:
        if os.path.isfile(file) and not file.endswith('.pkl'):
            remove(file)

    '''删除目录下的md.cfg和md.out文件'''
    for path in dirs:
        dump_nc = os.path.join(path, 'force.0.nc')
        single_md_out = os.path.join(path, 'md.out')
        single_md_cfg = os.path.join(path, 'md.cfg')
        gen_0_single_md_out = os.path.join(path, 'gen_0_md.out')
        gen_0_single_md_cfg = os.path.join(path, 'gen_0_md.cfg')
        remove(dump_nc)
        remove(single_md_out)
        remove(single_md_cfg)
        remove(gen_0_single_md_out)
        remove(gen_0_single_md_cfg)

def delete_database_data(train_path):
    files_to_delete = glob.glob(os.path.join(train_path, 'database*'))
    gen_0_files_to_delete = glob.glob(os.path.join(train_path, 'gen*'))
    for file in files_to_delete:
        if os.path.isfile(file) and not file.endswith('.pkl'):
            remove(file)
    for file in gen_0_files_to_delete:
        if os.path.isfile(file) and not file.endswith('.pkl'):
            remove(file)

def is_in_interval(x, interval):
    """判断 x 是否在区间 interval 内"""
    return interval[0] <= x < interval[1]

def find_interval(x, intervals, end_value,end_index):
    """使用二分查找找到 x 所在的区间"""
    # 提取区间的起始点
    starts = [interval[0] for interval in intervals]

    # 使用 bisect_left 找到 x 应该插入的位置
    index = bisect.bisect_left(starts, x)

    # 检查 x 是否在找到的区间内
    if index < len(intervals) and is_in_interval(x, intervals[index]):
        #print(f"Value {x} falls into interval {intervals[index]}")
        return index
        #return True
    elif index > 0 and is_in_interval(x, intervals[index - 1]):
        #print(f"Value {x} falls into interval {intervals[index-1]}")
        return index - 1
        #return True
    elif x == end_value:
        return end_index
    else:
        #print(f"Value {x} is outside the defined intervals")
        return None
        #return False

def process_data(args):
    stru_indexs, data_list, intervals = args
    #print(intervals)
    no_set_categories = [[] for _ in intervals]
    if len(intervals) != 0:
        end_value = intervals[-1][1]
        end_index = len(intervals) -1
        #print(intervals[-1])
        for stru_index, a in zip(stru_indexs, data_list):
            index = find_interval(a, intervals,end_value,end_index)
            if index is not None:
                no_set_categories[index].append(stru_index)
    return no_set_categories

# def process_data(args):
#     stru_indexs, data_list, intervals = args
#     print(intervals[-1])
#     no_set_categories = [[] for _ in intervals]
#     for stru_index, a in zip(stru_indexs, data_list):
#         for index, interval in enumerate(intervals):
#             if len(intervals) == index + 1:
#                 if interval[0] <= a <= interval[1]:
#                     no_set_categories[index].append(stru_index)
#             else:
#                 if interval[0] <= a < interval[1]:
#                     no_set_categories[index].append(stru_index)
#                     break
#     return no_set_categories

def parallel_process(stru_indexs, data_list, intervals):
    # 将数据分割成多个块
    num_processes = 5  # 可以根据你的CPU核心数量调整
    pool = Pool(processes=num_processes)
    #print(len(stru_indexs))
    if len(stru_indexs)<=num_processes:
        num_processes = len(stru_indexs)
    chunk_size = len(stru_indexs) // num_processes
    chunks = [stru_indexs[i:i + chunk_size] for i in range(0, len(stru_indexs), chunk_size)]
    data_list_chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建参数列表
    args_list = [(chunk, data_list_chunk, intervals) for chunk,data_list_chunk in zip(chunks,data_list_chunks)]
    # 并行处理
    results = pool.map(process_data, args_list)
    # 合并结果
    final_no_set_categories = [[] for _ in intervals]
    for result_chunk in results:
        for i, value in enumerate(result_chunk):
            final_no_set_categories[i].extend(value)
    return final_no_set_categories

def parallel_process(stru_indexs, data_list, intervals):
    return group_compact_indices(stru_indexs, data_list, intervals)


#####md#######
def freq_intervals_stru_cluster(data_list, stru_indexs, zero_freq_intervals, max_min, bin):
    if bin <= 0:
        return [], []
    span = float(max_min[0] - max_min[1])
    if span <= 0:
        return [], []
    bw = span / bin
    min_value = min(data_list)
    max_value = max(data_list)

    data_range = np.ptp(data_list)
    if data_range == 0 or bw <= 0:
        number_of_bins = 1
        new_bw = 0
    else:
        number_of_bins = max(1, int(np.ceil(data_range / bw)))
        new_bw = data_range / number_of_bins
    md_zero_freq_intervals = []
    if new_bw == 0:
        md_freq_intervals = [[min_value, max_value]]
    else:
        md_freq_intervals = [[min_value + i * new_bw, min_value + (i + 1) * new_bw] for i in range(number_of_bins)]

    temp = min_value + number_of_bins * new_bw
    if temp > max_value:
        md_freq_intervals[-1][1] = max_value

    for md in md_freq_intervals:
        if md[0] >= max_min[0] or md[1] <= max_min[1]:
            md_zero_freq_intervals.append(md)
        elif md[0] < max_min[1] and max_min[1] <= md[1] <= max_min[0]:
            md_zero_freq_intervals.append([md[0],max_min[1]])
        elif md[1] > max_min[0] and max_min[1] <= md[0] <= max_min[0]:
            md_zero_freq_intervals.append([max_min[0], md[1]])
    #print(md_zero_freq_intervals)
    #print(max_min)

    total_zero_freq_intervals = zero_freq_intervals + md_zero_freq_intervals
    total_zero_freq_intervals = sorted(total_zero_freq_intervals, key=lambda x: x[0])
    no_set_categories = parallel_process(stru_indexs, data_list, total_zero_freq_intervals)
    #print(len(no_set_categories))
    no_set_categories = [sublist for sublist in no_set_categories if sublist]
    categories = [set(a) for a in no_set_categories]

    return categories, no_set_categories

def md_sub_extract(array_data, stru_indexs, type_zero_freq_intervals_list,max_min, bins):
    array_data = numeric_data(array_data)
    D = array_data.shape[1]
    categories_list = []
    no_set_categories_list = []
    for i in range(D):
        new_data = column(array_data, i)
        categories, no_set_categories = freq_intervals_stru_cluster(new_data, stru_indexs, type_zero_freq_intervals_list[i], max_min[i], bins[i])
        categories_list.append(categories)
        no_set_categories_list.append(no_set_categories)
    return categories_list, no_set_categories_list

def md_extract(data, large_zero_freq_intervals_list, large_max_min, large_bins):
    md_data = ensure_decoded(data)
    D = 0
    for a in md_data:
        if len(a) != 0:
            D = values_and_indices(a)[0].shape[1]
            break
    large_categories_list = []
    large_no_set_categories_list = []

    for type_atoms, type_zero_freq_intervals_list, max_min, bins in zip(
            md_data, large_zero_freq_intervals_list, large_max_min, large_bins):

        if len(type_atoms):
            stru_temp, stru_index_temp = values_and_indices(type_atoms)
            if len(bins) < D:
                # The reference set holds no occupancy profile for this element, so
                # no candidate frame here can be covered by it: keep them all, each
                # as its own class, rather than indexing into the empty profile.
                keep = [[int(index)] for index in np.asarray(stru_index_temp).tolist()]
                categories_list = [[] for _ in range(D)]
                no_set_categories_list = [keep] + [[] for _ in range(D - 1)]
            else:
                categories_list, no_set_categories_list = md_sub_extract(
                    stru_temp, stru_index_temp, type_zero_freq_intervals_list, max_min, bins)
        else:
            categories_list = no_set_categories_list = [[] for _ in range(D)]

        large_categories_list.append(categories_list)
        large_no_set_categories_list.append(no_set_categories_list)

    # need_index_list = [category for categories in large_categories_list for category in categories]
    # no_set_need_index_list = [no_set_category for no_set_categories in large_no_set_categories_list for no_set_category in no_set_categories]
    need_index_list = []
    no_set_need_index_list = []
    for categories_list in large_categories_list:
        for categories in categories_list:
            need_index_list += categories
    for no_set_categories_list in large_no_set_categories_list:
        for no_set_categories in no_set_categories_list:
            no_set_need_index_list += no_set_categories

    return large_categories_list, large_no_set_categories_list, need_index_list, no_set_need_index_list

    #large_categories_list [[[], [], [{1600, 1673, 1041, 1076, 1688, 1759},{1},{2}], [], [], []], [[], [{1900}], [], [], [], []]] 维度[0,1,2]对应[type,D,{}/[]] {}/[]零频区间对应的结构index

@descriptor_stage
def main_sample_flow(
    pwd,
    dirs,
    dq_width,
    dq_width_method,
    dq_width_factor,
    body_list,
    ele,
    sort_ele,
    mtp_type,
    selection_budget_schedule,
    coverage_threshold_schedule,
    coverage_rate_method,
    logger,
    report_per_configuration_details=False,
    plateau_generations=None,
    min_coverage_delta=None,
    state_population=0,
    report_state_population_zero_baseline=False,
    mean_descriptor_enabled=False,
    mean_descriptor_state_population=0,
    npt_max_cell_volume_filter_factor=1.5,
):
    runtime_config = load_runtime_config(pwd)
    scheduler = build_scheduler_spec(runtime_config["scheduler"])
    runtime_parameter = runtime_config.get("parameter", {})
    encoding_cores = int(runtime_parameter.get("encoding_cores", 2))
    dimension_min_cover_workers = normalize_dimension_min_cover_workers(
        runtime_parameter.get("dimension_min_cover_workers", 0)
    )
    selection_budget_scope = normalize_selection_budget_scope(
        runtime_parameter.get("selection_budget_scope")
    )
    mean_descriptor_low_coverage_threshold = float(
        runtime_parameter.get("mean_descriptor_low_coverage_threshold", 90.0)
    )
    if not 0.0 <= mean_descriptor_low_coverage_threshold <= 100.0:
        raise ValueError("mean_descriptor_low_coverage_threshold must be between 0 and 100")

    method = dq_width_method

    ###编码###
    state_population = max(0, int(state_population))
    if isinstance(mean_descriptor_enabled, str):
        mean_descriptor_enabled = mean_descriptor_enabled.strip().lower() in {"true", "1", "yes", "on"}
    else:
        mean_descriptor_enabled = bool(mean_descriptor_enabled)
    mean_descriptor_state_population = max(0, int(mean_descriptor_state_population))
    mtp_path = os.path.join(pwd, SUS2_MODEL_DIR, 'current_0.mtp')
    train_cfg = os.path.join(pwd,'train_mlp','train.cfg')
    md_cfg = os.path.join(pwd, MD_WORK_DIR, 'md.cfg')
    data_out = os.path.join(pwd, 'train_mlp', 'database.out')
    md_out = os.path.join(pwd, MD_WORK_DIR, 'md.out')
    train_path = os.path.join(pwd, 'train_mlp')
    md_path = os.path.join(pwd, MD_WORK_DIR)
    plot_coverage_out_path = md_path
    num_ele = len(ele)
    gen_0_mtp = os.path.join(os.path.dirname(pwd), 'gen_0', SUS2_MODEL_DIR, 'current_0.mtp')

    gen_num = os.path.basename(pwd).replace('gen_', '')
    main_num = os.path.basename(os.path.dirname(pwd)).replace('main_', '')

    '''删除md相关文件,删除两遍的原因是，怕程序中断，数据会追加'''
    delete_md_data(md_path, dirs)

    '''体积截断：膨胀的MD帧在进入描述符/覆盖度/筛选之前就丢掉'''
    volume_guards = [
        npt_frame_volume_guard(case_dir, npt_max_cell_volume_filter_factor)
        for case_dir in dirs
    ]
    volume_stats = {}

    if int(gen_num) != 0 and int(main_num) != 0:
        # gen_0 与 md 必须用同一个 guard：两条编码的帧轴要逐帧对齐
        mul_encode(pwd, gen_0_mtp, dirs, 'gen_0_md.cfg', 'gen_0_md.out', scheduler.sus2_mlp_exe, scheduler.train_env, workers=encoding_cores, volume_guards=volume_guards)

    stage_progress('training_descriptor_encoding', input_path=train_cfg, workers=encoding_cores)
    encode_cfg_parallel(train_cfg, data_out, scheduler.sus2_mlp_exe, mtp_path, encoding_cores, scheduler.train_env)
    dirs_stru_counts = mul_encode(pwd, mtp_path, dirs, 'md.cfg', 'md.out', scheduler.sus2_mlp_exe, scheduler.train_env, workers=encoding_cores, volume_guards=volume_guards, volume_stats=volume_stats)
    volume_dropped_configurations = _report_volume_truncation(logger, pwd, volume_stats, npt_max_cell_volume_filter_factor)
    logger.info('External descriptor encoding completed; preparing numeric descriptor data.')

    ###cfg2xyz
    xyz_out_file_path = os.path.join(md_path, 'md.xyz')
    if os.path.exists(xyz_out_file_path):
        os.remove(xyz_out_file_path)
    cfg2xyz(ele, sort_ele, md_cfg, xyz_out_file_path)

    ###提取编码###
    body_name_list = body_list

    if int(gen_num) != 0 and int(main_num) != 0:
        #用gen_0的mtp势函数编码
        gen_0_data_out = os.path.join(pwd, 'train_mlp', 'gen_0_database.out')
        gen_0_md_out = os.path.join(md_path, 'gen_0_md.out')
        encode_cfg_parallel(train_cfg, gen_0_data_out, scheduler.sus2_mlp_exe, gen_0_mtp, encoding_cores, scheduler.train_env)
        gen0_train_store = build_descriptor_store(gen_0_data_out, 'gen_0_database', ele, mtp_type, mtp_path, body_name_list, train_path, mean_enabled=mean_descriptor_enabled)
        gen0_md_store = build_descriptor_store(gen_0_md_out, 'gen_0_md', ele, mtp_type, mtp_path, body_name_list, md_path, mean_enabled=mean_descriptor_enabled)

    train_store = build_descriptor_store(data_out, 'database', ele, mtp_type, mtp_path, body_name_list, train_path, mean_enabled=mean_descriptor_enabled)
    md_store = build_descriptor_store(md_out, 'md', ele, mtp_type, mtp_path, body_name_list, md_path, mean_enabled=mean_descriptor_enabled)

    large_need_index_list = []
    large_classes_num = []
    large_classes_stru_num = []
    large_min_cover_stru = []
    large_min_cover_stru_index = []
    large_type_coverage_rate = []
    large_zero_threshold_baseline_rate = []
    large_no_set_need_index_list = []
    ori_large_no_set_need_index_list = []
    large_dimension_tasks = []
    configuration_groups = build_configuration_groups(dirs, dirs_stru_counts)
    if not configuration_groups:
        raise RuntimeError(
            "No configuration groups were built from the sampled MD trajectories"
        )
    body_configuration_coverages = {}
    configuration_classes = {name: [] for name in configuration_groups}
    max_required_coverages = max_element_thresholds(coverage_threshold_schedule, num_ele)

    mean_select_index = []
    mean_convergence_result = {"converged": True}
    mean_coverage_rate = 100.0
    mean_configuration_coverages = {}
    if mean_descriptor_enabled:
        mean_select_index, mean_convergence_result, mean_coverage_rate, mean_configuration_coverages = mean_pre_sample_flow(
            pwd,
            dq_width,
            dq_width_method,
            dq_width_factor,
            selection_budget_schedule,
            coverage_threshold_schedule,
            coverage_rate_method,
            logger,
            element_count=num_ele,
            configuration_groups=configuration_groups,
            report_per_configuration_details=report_per_configuration_details,
            plateau_generations=plateau_generations,
            min_coverage_delta=min_coverage_delta,
            state_population=mean_descriptor_state_population,
            report_state_population_zero_baseline=False,
            mean_descriptor_low_coverage_threshold=mean_descriptor_low_coverage_threshold,
            dimension_min_cover_workers=dimension_min_cover_workers,
            elements=ele,
        )

    for body in body_list:
        stage_progress('coverage_and_candidates_' + body)
        body_dimension_tasks = []
        data_base_data = train_store.body(body)
        md_data = md_store.body(body)
        md_decoded = ensure_decoded(md_data)
        start = time.time()
        large_zero_freq_intervals_list, large_max_min, large_bins = data_base_distribution(
            data_base_data,
            dq_width,
            method,
            body,
            plot_model=False,
            dq_width_factor=dq_width_factor,
            state_population=state_population,
        )
        #print(large_zero_freq_intervals_list)
        end = time.time()
        print(f'body_{body}_data_base_distribution_time:', end - start)

        if int(gen_num) != 0 and int(main_num) != 0:
            data_temp = gen0_train_store.body(body)
            md_data_temp = gen0_md_store.body(body)
            a, b, c = data_base_distribution(
                data_temp,
                dq_width,
                method,
                body,
                plot_model=False,
                dq_width_factor=dq_width_factor,
                state_population=state_population,
            )
            coverage_reference_intervals = a
            coverage_reference_max_min = b
            coverage_source = ensure_decoded(md_data_temp)
        else:
            coverage_reference_intervals = large_zero_freq_intervals_list
            coverage_reference_max_min = large_max_min
            coverage_source = md_decoded

        zero_baseline_rate = None
        zero_baseline_detail = {}
        if report_state_population_zero_baseline:
            if int(gen_num) != 0 and int(main_num) != 0:
                zero_reference_intervals, zero_reference_max_min, _ = data_base_distribution(
                    data_temp,
                    dq_width,
                    method,
                    body,
                    plot_model=False,
                    dq_width_factor=dq_width_factor,
                    state_population=0,
                )
                zero_coverage_source = ensure_decoded(md_data_temp)
            else:
                zero_reference_intervals, zero_reference_max_min, _ = data_base_distribution(
                    data_base_data,
                    dq_width,
                    method,
                    body,
                    plot_model=False,
                    dq_width_factor=dq_width_factor,
                    state_population=0,
                )
                zero_coverage_source = md_decoded

        start = time.time()
        need_index_list = []
        no_set_need_index_list = []
        new_no_set_need_index_list = []
        config_coverages = []
        config_coverage_detail = {}
        zero_config_coverages = []
        for config_name, structure_indices in configuration_groups.items():
            config_md_data = slice_decoded_by_indices(md_decoded, structure_indices)
            _, config_no_set_index_list, config_need_index_list, config_no_set_need_index_list = md_extract(
                config_md_data,
                large_zero_freq_intervals_list,
                large_max_min,
                large_bins,
            )
            need_index_list.extend(config_need_index_list)
            no_set_need_index_list.extend(config_no_set_need_index_list)
            config_coverage_source = slice_decoded_by_indices(coverage_source, structure_indices)
            config_type_coverage_rate = coverage_rate(
                config_coverage_source,
                coverage_reference_intervals,
                coverage_reference_max_min,
                body,
                plot_coverage_out_path,
                coverage_rate_method,
                plot_model=False,
                plot_suffix=f'_{config_name}',
            )
            config_coverage_detail[config_name] = config_type_coverage_rate
            if report_state_population_zero_baseline:
                zero_rate = coverage_rate(
                    slice_decoded_by_indices(zero_coverage_source, structure_indices),
                    zero_reference_intervals,
                    zero_reference_max_min,
                    body,
                    plot_coverage_out_path,
                    coverage_rate_method,
                    plot_model=False,
                    plot_suffix=f'_{config_name}_threshold0_baseline',
                )
                zero_config_coverages.append([zero_rate])
                zero_baseline_detail[config_name] = zero_rate
            bool_result = [
                rate < max_required_coverages[index]
                for index, rate in enumerate(config_type_coverage_rate)
            ]
            if len(bool_result) != len(config_no_set_index_list):
                raise RuntimeError(
                    "Coverage element count does not match descriptor element count"
                )
            config_no_set_index_list = [
                value if enabled else []
                for value, enabled in zip(config_no_set_index_list, bool_result)
            ]
            if dimension_min_cover_workers != 0:
                config_tasks = (
                    build_dimension_tasks(
                        config_no_set_index_list,
                        body,
                        ele,
                        prefix=(str(config_name),),
                    )
                    if selection_budget_scope == "per_configuration"
                    else build_dimension_tasks(
                        config_no_set_index_list,
                        body,
                        ele,
                    )
                )
                body_dimension_tasks.extend(config_tasks)
            for no_set_categories_list in config_no_set_index_list:
                for no_set_categories in no_set_categories_list:
                    new_no_set_need_index_list += no_set_categories
                    configuration_classes[config_name].extend(no_set_categories)
            config_coverages.append([config_type_coverage_rate])
        type_coverage_rate = aggregate_element_coverages(config_coverages, num_ele) if config_coverages else [100.0] * num_ele
        if zero_config_coverages:
            zero_baseline_rate = aggregate_element_coverages(zero_config_coverages, num_ele)
        end = time.time()
        print(f'body_{body}_md_extract_and_type_coverage_rate_time:', end - start)
        if report_per_configuration_details:
            logger.info(
                f'per_configuration {body}-body coverage: '
                f'{summarize_configuration_coverages(config_coverage_detail)}'
            )
            if report_state_population_zero_baseline and zero_baseline_detail:
                logger.info(
                    f'per_configuration {body}-body coverage baseline '
                    f'(state_population=0): '
                    f'{summarize_configuration_coverages(zero_baseline_detail)}'
                )
        body_configuration_coverages[body] = dict(config_coverage_detail)

        large_need_index_list.append(need_index_list)
        large_classes_num.append(len(need_index_list))
        temp = set()
        for a in need_index_list:
            temp.update(a)
        large_classes_stru_num.append(len(temp))

        large_no_set_need_index_list.append(new_no_set_need_index_list)
        ori_large_no_set_need_index_list.append(no_set_need_index_list)

        if dimension_min_cover_workers == 0:
            tt = find_min_cover_set(no_set_need_index_list)
        else:
            tt = []
            large_dimension_tasks.extend(
                body_dimension_tasks
                if selection_budget_scope == "per_configuration"
                else merge_dimension_tasks(body_dimension_tasks)
            )

        large_min_cover_stru_index.append(tt)
        large_min_cover_stru.append(len(tt))
        large_type_coverage_rate.append(type_coverage_rate)
        if zero_baseline_rate is not None:
            large_zero_threshold_baseline_rate.append(zero_baseline_rate)

    split_min_cover_index = None
    if dimension_min_cover_workers != 0:
        split_min_cover_index, split_stats, selected_by_task = solve_dimension_tasks(
            large_dimension_tasks,
            dimension_min_cover_workers,
        )
        if (
            split_stats.get("task_count", 0) > 0
            and dimension_min_cover_workers == -1
            and split_stats.get("scheduler") is None
        ):
            logger.warning(
                "No scheduler allocation was detected; "
                "dimension_min_cover_workers=-1 uses affinity-visible CPUs."
            )
        split_min_cover_set = set(split_min_cover_index)
        for body_index, body in enumerate(body_list):
            body_selected = sorted(
                {
                    index
                    for key, selected in selected_by_task.items()
                    if key
                    and (
                        (selection_budget_scope == "per_configuration" and len(key) >= 2 and key[1] == str(body))
                        or (selection_budget_scope == "all_configurations" and key[0] == str(body))
                    )
                    for index in selected
                    if index in split_min_cover_set
                }
            )
            large_min_cover_stru_index[body_index] = body_selected
            large_min_cover_stru[body_index] = len(body_selected)

    per_configuration_aee_candidates = {}
    if selection_budget_scope == "per_configuration":
        for config_name in configuration_groups:
            config_lists = configuration_classes[config_name]
            if dimension_min_cover_workers == 0:
                config_min_cover = find_min_cover_set(config_lists)
            else:
                config_min_cover = sorted(
                    {
                        index
                        for key, selected in selected_by_task.items()
                        if key and key[0] == str(config_name)
                        for index in selected
                        if index in split_min_cover_set
                    }
                )
            per_configuration_aee_candidates[config_name] = rank_min_cover_indices(
                config_lists,
                config_min_cover,
            )

    '''每个body总计多少类'''
    f_1 = f'The number of classes:{large_classes_num}'
    '''每个body所有类中的结构数'''
    f_2 = f'The number of stru for all classes:{large_classes_stru_num}'
    '''每个body最小覆盖结构数'''
    f_3 = f'body_min_cover_stru:{large_min_cover_stru}'
    '''每个body，每个type覆盖率'''
    f_4 = f'type_coverage_rate:{[[round(float(num), 4) for num in row] for row in large_type_coverage_rate]}'
    f_5 = f'zero_threshold_baseline_type_coverage_rate:{[[round(float(num), 4) for num in row] for row in large_zero_threshold_baseline_rate]}'

    '''一个类中可以出现多个结构的index,每个结构累计频次多，说明结构越重要'''
    lists = list(itertools.chain(*large_no_set_need_index_list))
    print(f'num of ori_classes:{len(lists)}, num of current_classes (some type_atom have a weight of 0, delete these structure classes): {len(list(itertools.chain(*ori_large_no_set_need_index_list)))}')
    if dimension_min_cover_workers == 0:
        min_cover_index = find_min_cover_set(lists)
    else:
        min_cover_index = split_min_cover_index

    '''认为收敛的标准'''
    element_coverages = aggregate_element_coverages([large_type_coverage_rate], num_ele)
    real_stru_num, _, _ = determine_structure_budget(
        element_coverages,
        selection_budget_schedule,
        coverage_threshold_schedule,
    )
    if mean_descriptor_enabled:
        mean_real_stru_num, _, _ = determine_structure_budget(
            [mean_coverage_rate],
            selection_budget_schedule,
            scalar_thresholds_for_mean_descriptor(coverage_threshold_schedule, num_ele),
        )
        real_stru_num = max(real_stru_num, mean_real_stru_num)
    target = coverage_target_from_schedule(coverage_threshold_schedule)
    encoding_metric = min(element_coverages) if element_coverages else 100.0
    encoding_history = update_metric_history(
        convergence_history_path(pwd),
        "encoding_body_coverage",
        int(gen_num),
        encoding_metric,
    )
    encoding_convergence_result = evaluate_metric_convergence(
        encoding_history,
        target,
        plateau_generations=plateau_generations,
        min_coverage_delta=min_coverage_delta,
    )
    convergence = encoding_convergence_result["converged"] and mean_convergence_result["converged"]
    if convergence and volume_dropped_configurations:
        '''被体积截断的结构视为未收敛：下一代还要继续MD'''
        logger.info(
            'Sampling convergence held back: structures with truncated MD frames keep sampling: %s',
            ", ".join(sorted(volume_dropped_configurations)),
        )
        convergence = False

    # fw 挑出的每个结构，未覆盖原子环境数列表
    _, fw, select_index = fwss(lists, min_cover_index, real_stru_num)
    if len(min_cover_index) >= real_stru_num:
        new_tt = select_index
    else:
        new_tt = min_cover_index
    # print(new_tt)

    if selection_budget_scope == "per_configuration":
        total_select_index, _ = select_per_configuration_candidates(
            configuration_groups,
            body_configuration_coverages,
            body_list,
            per_configuration_aee_candidates,
            mean_select_index,
            mean_configuration_coverages,
            mean_descriptor_enabled,
            selection_budget_schedule,
            coverage_threshold_schedule,
            num_ele,
        )
        mean_selected_set = set(mean_select_index)
        aee_selected_set = {
            index
            for candidates in per_configuration_aee_candidates.values()
            for index in candidates
            if index in total_select_index
        }
        select_num_mean_pre_sample = len(mean_selected_set.intersection(total_select_index))
        select_num_AEE_sample = len(aee_selected_set)
        intersection = len(mean_selected_set.intersection(aee_selected_set))
    elif mean_descriptor_enabled:
        select_num_mean_pre_sample,select_num_AEE_sample, intersection, total_select_index = fwss_plus_mean_select_index(
            lists,
            min_cover_index,
            real_stru_num,
            mean_select_index,
            mean_coverage_rate,
            logger,
            mean_descriptor_low_coverage_threshold=mean_descriptor_low_coverage_threshold,
            apply_low_coverage_rule=False,
        )
    else:
        select_num_mean_pre_sample = 0
        select_num_AEE_sample = len(new_tt)
        intersection = 0
        total_select_index = new_tt

    '''生成stru_pkl文件，都是下一轮仍需采样的结构'''
    next_md_configuration_names, _ = _next_md_configurations(
        configuration_groups,
        mean_descriptor_enabled,
        mean_configuration_coverages,
        body_configuration_coverages,
        body_list,
        coverage_target_from_schedule(coverage_threshold_schedule),
        max_required_coverages,
    )
    if volume_dropped_configurations:
        '''膨胀过的结构即使覆盖度达标，也要进入下一轮MD'''
        forced_next_md_names = [
            name
            for name in sorted(volume_dropped_configurations)
            if name not in next_md_configuration_names
        ]
        if forced_next_md_names:
            logger.info(
                'Structures with truncated MD frames are kept for the next MD round: %s',
                ", ".join(forced_next_md_names),
            )
        next_md_configuration_names = list(next_md_configuration_names) + forced_next_md_names
    logger.info(f'per_configuration next MD seeds: {next_md_configuration_names}')
    mlp_return_strupkl(
        pwd,
        dirs,
        dirs_stru_counts,
        total_select_index,
        next_md_configuration_names=next_md_configuration_names,
    )


    if report_per_configuration_details:
        logger.info(
            f'per_configuration final selected_count: '
            f'{count_selected_by_configuration(total_select_index, configuration_groups)}'
        )
    else:
        process_name = 'AEE_sampling'
        '''出现结构的最大频次和最小频次'''
        try:
            logger.info(f'Complete the {process_name} process. {f_1} {f_2} {f_3} {f_4} max_min_freq: {[max(fw), min(fw)]}')
        except:
            logger.info(f'Complete the {process_name} process. {f_1} {f_2} {f_3} {f_4} max_min_freq: [NaN, NaN]')
        if report_state_population_zero_baseline:
            logger.info(f'{process_name} coverage baseline (state_population=0). {f_5}')
        ''' [最小覆盖结构数,从中筛选出来的结构]'''
        logger.info(f'num of select stru {[len(min_cover_index),len(select_index)]}')

    logger.info(f'num of select_num_mean_pre_sample:{select_num_mean_pre_sample}, num of select_num_AEE_sample:{select_num_AEE_sample}, num of repetition:{intersection}, total_num:{len(total_select_index)}')

    if convergence:
        log_plateau_convergence(
            logger,
            encoding_convergence_result,
            mean_convergence_result,
            mean_descriptor_enabled,
            plateau_generations,
            min_coverage_delta,
        )
        total_select_index = []

    stage_progress('selected_frame_output')
    atoms = SelectedFrames(xyz_out_file_path, total_select_index)
    select_atoms = []

    for index in total_select_index:
        select_atoms.append(atoms[index])
    select_xyz_path = os.path.join(md_path, f'{len(total_select_index)}_sample_filter.xyz')
    write(select_xyz_path,select_atoms,format='extxyz')

    delete_md_data(md_path, dirs)

    '''测试的时候可以保留，不删除数据'''
    delete_database_data(train_path)

    return new_tt, fw, select_atoms

if __name__ == '__main__':
    dq_width = 0.01
    dq_width_method = 'Freedman_Diaconis'
    body_list = ['two']
    hyx_mtp_path = 'hyx.mtp'
    mtp_type = 'l2k2'
    ele = ['Al','As','Ga']
    sort_ele = True
    selection_budget_schedule = 30
    coverage_threshold_schedule = [99.5, 99.9, 99.95]

    tt, fw, select_atoms = main_sample_flow(
        pwd,
        dirs,
        dq_width,
        dq_width_method,
        1.0,
        body_list,
        ele,
        sort_ele,
        mtp_type,
        selection_budget_schedule,
        coverage_threshold_schedule,
        coverage_rate_method,
        logger,
    )
    print(tt)
    print(len(tt))

    # bw = 0.001
    # # method = 'self_input'
    # # method = 'scott'
    # method = 'Freedman_Diaconis'
    # # method = 'std'
    # body_list = ['two']
    # plot_model = False
    # hyx_mtp_path = 'hyx.mtp'
    # mtp_type = 'l2k2'
    # ele = ['O', '1']
    # des_out2pkl('md.out', 'md', ele, mtp_type, hyx_mtp_path,body_list)
    # des_out2pkl('database.out', 'database', ele, mtp_type, hyx_mtp_path,body_list)
    # large_need_index_list = []
    # for body in body_list:
    #     data_base_data = f"database_{body}_body_coding_zlib.pkl"
    #     md_data = f"md_{body}_body_coding_zlib.pkl"
    #     large_zero_freq_intervals_list, large_max_min, large_bins = data_base_distribution(data_base_data, bw, method,
    #                                                                                        plot_model=False)
    #
    #     # for type_max_min,type_large_bins in zip(large_max_min, large_bins):
    #     #     for max_min,bins in zip(type_max_min,type_large_bins):
    #     #         print(max_min,bins)
    #     index_list, need_index_list = md_extract(md_data, large_zero_freq_intervals_list, large_max_min, large_bins)
    #     print(f'The number of classes:{len(need_index_list)}')
    #
    #     large_need_index_list +=need_index_list
    #     temp = []
    #     for a in need_index_list:
    #         temp = temp + list(a)
    #     print(f'The number of stru for all classes:{len(list(set(temp)))}')
    #     type_coverage_rate = coverage_rate(md_data, large_zero_freq_intervals_list, large_max_min, body)
    #     tt = find_min_cover_set(need_index_list)
    #     print(f'min_cover_stru:{len(tt)}')
    #     print(f'type_coverage_rate:{type_coverage_rate}')
    # tt = find_min_cover_set(large_need_index_list)
    # print(len(tt))




