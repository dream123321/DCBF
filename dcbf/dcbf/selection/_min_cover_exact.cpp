#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ElementInfo {
    long long value;
    long long flat_count;
    long long first_seen;
};

static PyObject* exact_min_cover_ints(PyObject* self, PyObject* args) {
    PyObject* input = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return nullptr;
    }

    PyObject* outer_seq = PySequence_Fast(input, "lists must be a sequence");
    if (outer_seq == nullptr) {
        return nullptr;
    }

    const Py_ssize_t set_count = PySequence_Fast_GET_SIZE(outer_seq);
    std::vector<std::vector<long long>> unique_sets;
    unique_sets.reserve(static_cast<size_t>(set_count));

    std::unordered_map<long long, long long> flat_counts;
    std::unordered_map<long long, long long> first_seen;
    long long first_seen_counter = 0;

    for (Py_ssize_t i = 0; i < set_count; ++i) {
        PyObject* subset_obj = PySequence_Fast_GET_ITEM(outer_seq, i);
        PyObject* subset_seq = PySequence_Fast(subset_obj, "each subset must be a sequence");
        if (subset_seq == nullptr) {
            Py_DECREF(outer_seq);
            return nullptr;
        }

        std::unordered_set<long long> dedup;
        std::vector<long long> unique_values;
        const Py_ssize_t subset_len = PySequence_Fast_GET_SIZE(subset_seq);
        unique_values.reserve(static_cast<size_t>(subset_len));

        for (Py_ssize_t j = 0; j < subset_len; ++j) {
            PyObject* item = PySequence_Fast_GET_ITEM(subset_seq, j);
            long long value = PyLong_AsLongLong(item);
            if (PyErr_Occurred()) {
                Py_DECREF(subset_seq);
                Py_DECREF(outer_seq);
                return nullptr;
            }
            flat_counts[value] += 1;
            if (first_seen.find(value) == first_seen.end()) {
                first_seen[value] = first_seen_counter++;
            }
            if (dedup.insert(value).second) {
                unique_values.push_back(value);
            }
        }

        unique_sets.push_back(std::move(unique_values));
        Py_DECREF(subset_seq);
    }

    Py_DECREF(outer_seq);

    if (unique_sets.empty()) {
        return PyList_New(0);
    }

    std::vector<ElementInfo> elements;
    elements.reserve(flat_counts.size());
    for (const auto& entry : flat_counts) {
        elements.push_back(ElementInfo{entry.first, entry.second, first_seen[entry.first]});
    }

    std::sort(elements.begin(), elements.end(), [](const ElementInfo& left, const ElementInfo& right) {
        if (left.flat_count != right.flat_count) {
            return left.flat_count > right.flat_count;
        }
        return left.first_seen < right.first_seen;
    });

    std::unordered_map<long long, int> rank_by_value;
    rank_by_value.reserve(elements.size());
    for (size_t rank = 0; rank < elements.size(); ++rank) {
        rank_by_value[elements[rank].value] = static_cast<int>(rank);
    }

    std::vector<std::vector<int>> set_to_ranks(unique_sets.size());
    std::vector<std::vector<int>> element_to_sets(elements.size());
    for (size_t set_index = 0; set_index < unique_sets.size(); ++set_index) {
        auto& ranks = set_to_ranks[set_index];
        ranks.reserve(unique_sets[set_index].size());
        for (const long long value : unique_sets[set_index]) {
            int rank = rank_by_value[value];
            ranks.push_back(rank);
            element_to_sets[rank].push_back(static_cast<int>(set_index));
        }
    }

    std::vector<int> current_count(elements.size(), 0);
    std::vector<unsigned char> active(elements.size(), 0);
    std::vector<unsigned char> remaining(unique_sets.size(), 1);
    int remaining_count = static_cast<int>(unique_sets.size());

    using MinHeap = std::priority_queue<int, std::vector<int>, std::greater<int>>;
    std::vector<MinHeap> buckets(unique_sets.size() + 1);
    int max_count = 0;
    for (size_t rank = 0; rank < element_to_sets.size(); ++rank) {
        int count = static_cast<int>(element_to_sets[rank].size());
        current_count[rank] = count;
        if (count > 0) {
            active[rank] = 1;
            buckets[count].push(static_cast<int>(rank));
            if (count > max_count) {
                max_count = count;
            }
        }
    }

    std::vector<long long> cover;
    cover.reserve(elements.size());

    while (remaining_count > 0 && max_count > 0) {
        while (max_count > 0) {
            auto& bucket = buckets[max_count];
            while (!bucket.empty()) {
                int rank = bucket.top();
                if (!active[rank] || current_count[rank] != max_count) {
                    bucket.pop();
                    continue;
                }
                break;
            }
            if (!bucket.empty()) {
                break;
            }
            --max_count;
        }

        if (max_count == 0) {
            break;
        }

        auto& best_bucket = buckets[max_count];
        int best_rank = best_bucket.top();
        best_bucket.pop();
        if (!active[best_rank] || current_count[best_rank] != max_count) {
            continue;
        }

        cover.push_back(elements[best_rank].value);
        for (const int set_index : element_to_sets[best_rank]) {
            if (!remaining[set_index]) {
                continue;
            }
            remaining[set_index] = 0;
            --remaining_count;
            for (const int rank : set_to_ranks[set_index]) {
                if (!active[rank]) {
                    continue;
                }
                const int new_count = current_count[rank] - 1;
                current_count[rank] = new_count;
                if (new_count == 0) {
                    active[rank] = 0;
                } else {
                    buckets[new_count].push(rank);
                }
            }
        }
    }

    PyObject* cover_set = PySet_New(nullptr);
    if (cover_set == nullptr) {
        return nullptr;
    }
    for (const long long item : cover) {
        PyObject* value = PyLong_FromLongLong(item);
        if (value == nullptr) {
            Py_DECREF(cover_set);
            return nullptr;
        }
        if (PySet_Add(cover_set, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(cover_set);
            return nullptr;
        }
        Py_DECREF(value);
    }
    PyObject* output = PySequence_List(cover_set);
    Py_DECREF(cover_set);
    return output;
}

static PyMethodDef module_methods[] = {
    {"exact_min_cover_ints", exact_min_cover_ints, METH_VARARGS, "Exact legacy-compatible greedy min-cover for integer lists."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_min_cover_exact",
    "Exact legacy-compatible min-cover backend.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__min_cover_exact(void) {
    return PyModule_Create(&module_definition);
}
