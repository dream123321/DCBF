import math
import random
import time

from .selection_core import frequency_counter, greedy_cover_elements


def cover_count(sets, elem):
    return sum(1 for item_set in sets if elem in item_set)


def sort_by_count_with_counter(lists):
    counter = frequency_counter(lists)
    return sorted(counter, key=counter.get, reverse=True)


def list2intersection(list1, list2):
    return set(list1).intersection(set(list2))


def set_intersection(list1, list2):
    return len(list2intersection(list1, list2))


def find_min_cover_set(lists):
    return greedy_cover_elements(lists)


def fwss(lists, min_cover_index, num):
    counter = frequency_counter(lists)
    sorted_min_cover_index = sorted(min_cover_index, key=lambda value: counter[value], reverse=True)
    count_list = [counter[value] for value in sorted_min_cover_index]
    aa = num // 5
    bb = num - aa
    if bb == 0:
        bb = 1
    cc = (len(sorted_min_cover_index) - aa) // bb
    if cc == 0:
        cc = 1
    print(f"a={aa};b={bb};c={cc}")
    select_sorted_min_cover_index = sorted_min_cover_index[:aa] + sorted_min_cover_index[aa::cc]
    if num >= len(sorted_min_cover_index):
        select_sorted_min_cover_index = sorted_min_cover_index[:num]
    return sorted_min_cover_index, count_list, select_sorted_min_cover_index


def fwss_plus_mean_select_index(lists, min_cover_index, num, mean_select_index, mean_coverage_rate, logger):
    counter = frequency_counter(lists)
    sorted_min_cover_index = sorted(min_cover_index, key=lambda value: counter[value], reverse=True)
    temp_sorted_min_cover_index = sorted_min_cover_index
    sorted_min_cover_index = mean_select_index + [
        item for item in sorted_min_cover_index if item not in mean_select_index
    ]

    set_mean_coverage_rate_value = 90

    if mean_coverage_rate > set_mean_coverage_rate_value:
        aa = num // 5
        bb = num - aa
        if bb == 0:
            bb = 1
        cc = (len(sorted_min_cover_index) - aa) // bb
        if cc == 0:
            cc = 1
        print(f"a={aa};b={bb};c={cc}")
        select_sorted_min_cover_index = sorted_min_cover_index[:aa] + sorted_min_cover_index[aa::cc]
        if num >= len(sorted_min_cover_index):
            select_sorted_min_cover_index = sorted_min_cover_index[:num]
    else:
        if len(sorted_min_cover_index) * 0.2 < num:
            select_num = math.floor(len(sorted_min_cover_index) * 0.8)
            select_sorted_min_cover_index = sorted_min_cover_index[select_num:]
        else:
            select_sorted_min_cover_index = sorted_min_cover_index[len(sorted_min_cover_index) - num:]
        logger.info(
            f"mean_coverage_rate is less than {set_mean_coverage_rate_value}%, select the last 20% structure"
        )

    return (
        set_intersection(select_sorted_min_cover_index, mean_select_index),
        set_intersection(select_sorted_min_cover_index, temp_sorted_min_cover_index),
        set_intersection(
            list2intersection(select_sorted_min_cover_index, mean_select_index),
            list2intersection(select_sorted_min_cover_index, temp_sorted_min_cover_index),
        ),
        select_sorted_min_cover_index,
    )


def generate_random_sets(num_sets, min_set_size, max_set_size, value_range):
    sets = []
    for _ in range(num_sets):
        set_size = random.randint(min_set_size, max_set_size)
        random_set = set(random.sample(range(value_range), set_size))
        sets.append(random_set)
    return sets


if __name__ == "__main__":
    num_sets = 2000
    min_set_size = 1
    max_set_size = 5000
    value_range = 50000

    random.seed(42)
    class_sets = generate_random_sets(num_sets, min_set_size, max_set_size, value_range)

    start = time.time()
    result = find_min_cover_set(class_sets)
    end = time.time()
    print(f"time:{end - start}, min_cover:{result}")

    start = time.time()
    print(fwss(class_sets, result, 10))
    end = time.time()
    print("time:", end - start)
