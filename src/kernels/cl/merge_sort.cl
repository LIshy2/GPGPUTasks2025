#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);
    int block_i = i / (sorted_k * 2);
    int lb = block_i * sorted_k * 2;
    int rb = block_i * sorted_k * 2 + sorted_k;

    if (rb >= n) {
        return;
    }

    int d = i - block_i * sorted_k * 2;

    int l = max(0, d - sorted_k);
    int r = min(d, sorted_k);

    while (l < r) {
        int m = (l + r) / 2;
        int li = m;
        int ri = d - m;
        if (li < sorted_k && (input_data[lb + li] < input_data[rb + ri - 1] || ri >= sorted_k || rb + ri - 1 >= n)) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (rb + d - l >= n || input_data[lb + l] <= input_data[rb + d - l]) {
        output_data[i] = input_data[lb + l];
    } else {
        output_data[i] = input_data[rb + d - l];
    }
}
