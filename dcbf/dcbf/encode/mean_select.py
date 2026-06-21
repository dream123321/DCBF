import os
import numpy as np
from .coverage_rate import coverage_rate
from .coverage_policy import (
    aggregate_element_coverages,
    count_selected_by_configuration,
    determine_structure_budget,
    ensure_decoded,
    scalar_thresholds_for_mean_descriptor,
    slice_decoded_by_indices,
    summarize_configuration_coverages,
)
from .convergence_control import (
    convergence_history_path,
    coverage_target_from_schedule,
    evaluate_metric_convergence,
    update_metric_history,
)
from .find_min_cover_set import find_min_cover_set,fwss
import itertools
from .data_distri import data_base_distribution
from .mlp_encoding_extract import decode
from .selection_core import group_structure_indices_by_interval
from ..path_names import MD_WORK_DIR

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
    return group_structure_indices_by_interval(stru_indexs, data_list, intervals)


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
    D = len(array_data[0])
    categories_list = []
    no_set_categories_list = []
    array_data = np.array(array_data)
    for i in range(D):
        new_data = array_data[:, i]
        categories, no_set_categories = freq_intervals_stru_cluster(new_data, stru_indexs, type_zero_freq_intervals_list[i], max_min[i], bins[i])
        categories_list.append(categories)
        no_set_categories_list.append(no_set_categories)
    return categories_list, no_set_categories_list

def md_extract(data, large_zero_freq_intervals_list, large_max_min, large_bins):
    md_data = ensure_decoded(data)
    D = 0
    for a in md_data:
        if len(a) != 0:
            D = np.array(a).shape[1] - 1
            break
    large_categories_list = []
    large_no_set_categories_list = []

    for type_atoms, type_zero_freq_intervals_list, max_min, bins in zip(
            md_data, large_zero_freq_intervals_list, large_max_min, large_bins):

        if type_atoms:
            stru_temp = [atom[:-1] for atom in type_atoms]
            stru_index_temp = [atom[-1] for atom in type_atoms]
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

    return large_categories_list, need_index_list, no_set_need_index_list

def mean_pre_sample_flow(
    pwd,
    dq_width,
    dq_width_method,
    dq_width_factor,
    selection_budget_schedule,
    coverage_threshold_schedule,
    coverage_rate_method,
    logger,
    element_count=1,
    configuration_groups=None,
    coverage_calculation_mode='global',
    report_per_configuration_details=False,
    plateau_generations=None,
    min_coverage_delta=None,
    state_population=0,
    report_state_population_zero_baseline=False,
):
    train_path = os.path.join(pwd, 'train_mlp')
    md_path = os.path.join(pwd, MD_WORK_DIR)
    plot_coverage_out_path = md_path
    state_population = max(0, int(state_population))

    gen_num = os.path.basename(pwd).replace('gen_', '')
    main_num = os.path.basename(os.path.dirname(pwd)).replace('main_', '')

    large_need_index_list = []
    large_classes_num = []
    large_classes_stru_num = []
    large_min_cover_stru = []
    large_min_cover_stru_index = []
    large_type_coverage_rate = []
    large_no_set_need_index_list = []

    body_list = ['two']
    for body in body_list:
        data_base_data = os.path.join(train_path, f"database_mean_coding_zlib.pkl")
        md_data = os.path.join(md_path, f"md_mean_coding_zlib.pkl")
        md_decoded = ensure_decoded(md_data)

        large_zero_freq_intervals_list, large_max_min, large_bins = data_base_distribution(
            data_base_data,
            dq_width,
            dq_width_method,
            body,
            plot_model=False,
            dq_width_factor=dq_width_factor,
            state_population=state_population,
        )
        if int(gen_num) != 0 and int(main_num) != 0:
            data_temp = os.path.join(train_path, f"gen_0_database_mean_coding_zlib.pkl")
            md_data_temp = os.path.join(md_path, f"gen_0_md_mean_coding_zlib.pkl")
            a, b, c = data_base_distribution(
                data_temp,
                dq_width,
                dq_width_method,
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

        if coverage_calculation_mode == 'per_configuration' and configuration_groups:
            need_index_list = []
            no_set_need_index_list = []
            config_coverages = []
            config_coverage_detail = {}
            for config_name, structure_indices in configuration_groups.items():
                config_md_data = slice_decoded_by_indices(md_decoded, structure_indices)
                _, config_need_index_list, config_no_set_need_index_list = md_extract(
                    config_md_data,
                    large_zero_freq_intervals_list,
                    large_max_min,
                    large_bins,
                )
                need_index_list.extend(config_need_index_list)
                no_set_need_index_list.extend(config_no_set_need_index_list)
                config_coverages.append(
                    coverage_rate(
                        slice_decoded_by_indices(coverage_source, structure_indices),
                        coverage_reference_intervals,
                        coverage_reference_max_min,
                        body,
                        plot_coverage_out_path,
                        coverage_rate_method,
                        plot_model=False,
                        plot_suffix=f'_{config_name}',
                    )
                )
                config_coverage_detail[config_name] = config_coverages[-1]
            type_coverage_rate = aggregate_element_coverages([[rates] for rates in config_coverages], len(config_coverages[0])) if config_coverages else [100.0]
        else:
            _, need_index_list, no_set_need_index_list = md_extract(md_decoded, large_zero_freq_intervals_list, large_max_min, large_bins)
            type_coverage_rate = coverage_rate(
                coverage_source,
                coverage_reference_intervals,
                coverage_reference_max_min,
                body,
                plot_coverage_out_path,
                coverage_rate_method,
                plot_model=False,
            )

        large_need_index_list.append(need_index_list)
        large_no_set_need_index_list.append(no_set_need_index_list)
        large_classes_num.append(len(need_index_list))
        temp = []
        for a in need_index_list:
            temp = temp + list(a)
        large_classes_stru_num.append(len(list(set(temp))))

        tt = find_min_cover_set(no_set_need_index_list)

        large_min_cover_stru_index.append(tt)
        large_min_cover_stru.append(len(tt))
        large_type_coverage_rate.append(type_coverage_rate)

    '''每个body总计多少类'''
    f_1 = f'The number of classes:{large_classes_num}'
    '''每个body所有类中的结构数'''
    f_2 = f'The number of stru for all classes:{large_classes_stru_num}'
    '''每个body最小覆盖结构数'''
    f_3 = f'body_min_cover_stru:{large_min_cover_stru}'
    '''每个body，每个type覆盖率'''
    f_4 = f'type_coverage_rate:{[[round(float(num), 4) for num in row] for row in large_type_coverage_rate]}'

    '''一个类中可以出现多个结构的index,每个结构累计频次多，说明结构越重要'''
    lists = list(itertools.chain(*large_no_set_need_index_list))
    min_cover_index = find_min_cover_set(lists)

    '''认为收敛的标准'''
    mean_thresholds = scalar_thresholds_for_mean_descriptor(coverage_threshold_schedule, element_count)
    min_coverage_rate = min(np.array(large_type_coverage_rate).flatten())
    real_stru_num, _, _ = determine_structure_budget([min_coverage_rate], selection_budget_schedule, mean_thresholds)
    target = coverage_target_from_schedule(coverage_threshold_schedule)
    metric_history = update_metric_history(
        convergence_history_path(pwd),
        "mean_descriptor",
        int(gen_num),
        min_coverage_rate,
    )
    convergence_result = evaluate_metric_convergence(
        metric_history,
        target,
        plateau_generations=plateau_generations,
        min_coverage_delta=min_coverage_delta,
    )

    # fw 挑出的每个结构，未覆盖原子环境数列表
    _, fw, select_index = fwss(lists, min_cover_index, real_stru_num)
    if len(min_cover_index) >= real_stru_num:
        new_tt = select_index
    else:
        new_tt = min_cover_index

    if convergence_result["converged"]:
        new_tt = []

    if report_per_configuration_details and coverage_calculation_mode == 'per_configuration' and configuration_groups:
        logger.info(
            f'per_configuration mean_descriptor coverage: '
            f'{summarize_configuration_coverages(config_coverage_detail)}'
        )
        logger.info(
            f'per_configuration mean_descriptor selected_count: '
            f'{count_selected_by_configuration(new_tt, configuration_groups)}'
        )
    else:
        '''出现结构的最大频次和最小频次'''
        try:
            logger.info(f'mean_descriptor pre-sampling process. {f_1} {f_2} {f_3} {f_4} max_min_freq: {[max(fw), min(fw)]}')
        except:
            logger.info(f'mean_descriptor pre-sampling process. {f_1} {f_2} {f_3} {f_4} max_min_freq: [NaN, NaN]')
        ''' [最小覆盖结构数,从中筛选出来的结构]'''
        logger.info(f'num of select stru {[len(min_cover_index),len(select_index)]}')

    return new_tt,convergence_result,min_coverage_rate
