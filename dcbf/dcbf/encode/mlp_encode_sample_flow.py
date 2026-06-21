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
from .mlp_encoding_extract import decode,des_out2pkl
from .mean_encoding_extract import mean_des_out2pkl
import time
from .coverage_rate import coverage_rate
from .find_min_cover_set import find_min_cover_set,fwss,fwss_plus_mean_select_index
import itertools
from .file_conversion import dump2cfg, cfg2xyz, remove
from .mlp_mul_encode import mul_encode
from .cfg_descriptor_encode import encode_cfg_parallel
from .mlp_return_strupkl import mlp_return_strupkl
import glob
from .data_distri import data_base_distribution
from .mean_select import mean_pre_sample_flow
from .selection_core import group_structure_indices_by_interval
from ..path_names import MD_WORK_DIR, SUS2_MODEL_DIR
from ..runtime_config import build_scheduler_spec, load_runtime_config
import sys

def delete_md_data(md_path,dirs):
    '''删除md相关文件'''
    files_to_delete = glob.glob(os.path.join(md_path, 'md*'))
    gen_0_files_to_delete = glob.glob(os.path.join(md_path, 'gen*'))
    for file in files_to_delete:
        remove(file)
    for file in gen_0_files_to_delete:
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
        remove(file)
    for file in gen_0_files_to_delete:
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

    return large_categories_list, large_no_set_categories_list, need_index_list, no_set_need_index_list

    #large_categories_list [[[], [], [{1600, 1673, 1041, 1076, 1688, 1759},{1},{2}], [], [], []], [[], [{1900}], [], [], [], []]] 维度[0,1,2]对应[type,D,{}/[]] {}/[]零频区间对应的结构index

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
    coverage_calculation_mode='global',
    report_per_configuration_details=False,
    plateau_generations=None,
    min_coverage_delta=None,
    state_population=0,
    report_state_population_zero_baseline=False,
    mean_descriptor_enabled=False,
    mean_descriptor_state_population=0,
):
    runtime_config = load_runtime_config(pwd)
    scheduler = build_scheduler_spec(runtime_config["scheduler"])
    encoding_cores = int(runtime_config.get("parameter", {}).get("encoding_cores", 2))

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

    if int(gen_num) != 0 and int(main_num) != 0:
        mul_encode(pwd, gen_0_mtp, dirs, 'gen_0_md.cfg', 'gen_0_md.out', scheduler.sus2_mlp_exe, scheduler.train_env)

    encode_cfg_parallel(train_cfg, data_out, scheduler.sus2_mlp_exe, mtp_path, encoding_cores, scheduler.train_env)
    dirs_stru_counts = mul_encode(pwd, mtp_path, dirs, 'md.cfg', 'md.out', scheduler.sus2_mlp_exe, scheduler.train_env)
    logger.info(f'Complete coding process.')

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
        des_out2pkl(gen_0_data_out, 'gen_0_database', num_ele, mtp_type, mtp_path, body_name_list, train_path)
        des_out2pkl(gen_0_md_out, 'gen_0_md', num_ele, mtp_type, mtp_path, body_name_list, md_path)
        if mean_descriptor_enabled:
            mean_des_out2pkl(gen_0_data_out, 'gen_0_database', num_ele, mtp_type, mtp_path, body_name_list, train_path)
            mean_des_out2pkl(gen_0_md_out, 'gen_0_md', num_ele, mtp_type, mtp_path, body_name_list, md_path)

    des_out2pkl(data_out, 'database', num_ele, mtp_type, mtp_path, body_name_list,train_path)
    des_out2pkl(md_out, 'md', num_ele, mtp_type, mtp_path, body_name_list,md_path)
    if mean_descriptor_enabled:
        mean_des_out2pkl(data_out, 'database', num_ele, mtp_type, mtp_path, body_name_list, train_path)
        mean_des_out2pkl(md_out, 'md', num_ele, mtp_type, mtp_path, body_name_list, md_path)

    large_need_index_list = []
    large_classes_num = []
    large_classes_stru_num = []
    large_min_cover_stru = []
    large_min_cover_stru_index = []
    large_type_coverage_rate = []
    large_zero_threshold_baseline_rate = []
    large_no_set_need_index_list = []
    ori_large_no_set_need_index_list = []
    configuration_groups = None
    if coverage_calculation_mode == 'per_configuration':
        configuration_groups = build_configuration_groups(dirs, dirs_stru_counts)
    max_required_coverages = max_element_thresholds(coverage_threshold_schedule, num_ele)

    mean_select_index = []
    mean_convergence_result = {"converged": True}
    mean_coverage_rate = 100.0
    if mean_descriptor_enabled:
        mean_select_index, mean_convergence_result, mean_coverage_rate = mean_pre_sample_flow(
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
            coverage_calculation_mode=coverage_calculation_mode,
            report_per_configuration_details=report_per_configuration_details,
            plateau_generations=plateau_generations,
            min_coverage_delta=min_coverage_delta,
            state_population=mean_descriptor_state_population,
            report_state_population_zero_baseline=False,
        )

    for body in body_list:
        data_base_data = os.path.join(train_path, f"database_{body}_body_coding_zlib.pkl")
        md_data = os.path.join(md_path, f"md_{body}_body_coding_zlib.pkl")
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
            data_temp = os.path.join(train_path, f"gen_0_database_{body}_body_coding_zlib.pkl")
            md_data_temp = os.path.join(md_path, f"gen_0_md_{body}_body_coding_zlib.pkl")
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

        if coverage_calculation_mode == 'per_configuration' and configuration_groups:
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
                bool_result = [rate < max_required_coverages[index] for index, rate in enumerate(config_type_coverage_rate)]
                if len(bool_result) != len(config_no_set_index_list):
                    print('len(bool_result) != len(no_set_index_list)! please check!')
                    sys.exit(1)
                config_no_set_index_list = [value if enabled else [] for value, enabled in zip(config_no_set_index_list, bool_result)]
                for no_set_categories_list in config_no_set_index_list:
                    for no_set_categories in no_set_categories_list:
                        new_no_set_need_index_list += no_set_categories
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
        else:
            start = time.time()
            _, no_set_index_list, need_index_list, no_set_need_index_list = md_extract(
                md_decoded,
                large_zero_freq_intervals_list,
                large_max_min,
                large_bins,
            )
            end = time.time()
            print(f'body_{body}_md_extract_time:',end-start)

            start = time.time()
            type_coverage_rate = coverage_rate(
                coverage_source,
                coverage_reference_intervals,
                coverage_reference_max_min,
                body,
                plot_coverage_out_path,
                coverage_rate_method,
                plot_model=False,
            )
            if report_state_population_zero_baseline:
                zero_baseline_rate = coverage_rate(
                    zero_coverage_source,
                    zero_reference_intervals,
                    zero_reference_max_min,
                    body,
                    plot_coverage_out_path,
                    coverage_rate_method,
                    plot_model=False,
                    plot_suffix='_threshold0_baseline',
                )
            end = time.time()
            print(f'body_{body}_type_coverage_rate_time:',end-start)

            bool_result = [rate < max_required_coverages[index] for index, rate in enumerate(type_coverage_rate)]
            if len(bool_result) != len(no_set_index_list):
                print('len(bool_result) != len(no_set_index_list)! please check!')
                sys.exit(1)

            no_set_index_list = [value if enabled else [] for value, enabled in zip(no_set_index_list, bool_result)]
            new_no_set_need_index_list = []
            for no_set_categories_list in no_set_index_list:
                for no_set_categories in no_set_categories_list:
                    new_no_set_need_index_list += no_set_categories

        large_need_index_list.append(need_index_list)
        large_classes_num.append(len(need_index_list))
        temp = []
        for a in need_index_list:
            temp = temp + list(a)
        large_classes_stru_num.append(len(list(set(temp))))

        large_no_set_need_index_list.append(new_no_set_need_index_list)
        ori_large_no_set_need_index_list.append(no_set_need_index_list)

        tt = find_min_cover_set(no_set_need_index_list)

        large_min_cover_stru_index.append(tt)
        large_min_cover_stru.append(len(tt))
        large_type_coverage_rate.append(type_coverage_rate)
        if zero_baseline_rate is not None:
            large_zero_threshold_baseline_rate.append(zero_baseline_rate)

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
    min_cover_index = find_min_cover_set(lists)

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

    # fw 挑出的每个结构，未覆盖原子环境数列表
    _, fw, select_index = fwss(lists, min_cover_index, real_stru_num)
    if len(min_cover_index) >= real_stru_num:
        new_tt = select_index
    else:
        new_tt = min_cover_index
    atoms = list(iread(xyz_out_file_path))
    # print(new_tt)

    if mean_descriptor_enabled:
        select_num_mean_pre_sample,select_num_AEE_sample, intersection, total_select_index = fwss_plus_mean_select_index(lists, min_cover_index, real_stru_num, mean_select_index,mean_coverage_rate,logger)
    else:
        select_num_mean_pre_sample = 0
        select_num_AEE_sample = len(new_tt)
        intersection = 0
        total_select_index = new_tt

    '''生成stru_pkl文件，都是不收敛的结构'''
    mlp_return_strupkl(pwd, dirs, dirs_stru_counts, total_select_index)


    if report_per_configuration_details and coverage_calculation_mode == 'per_configuration' and configuration_groups:
        logger.info(
            f'per_configuration final selected_count: '
            f'{count_selected_by_configuration(total_select_index, configuration_groups)}'
        )
    else:
        process_name = 'AEE_sampling' if coverage_calculation_mode == 'per_configuration' else 'ocbf_sampling'
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
        total_select_index = []
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









