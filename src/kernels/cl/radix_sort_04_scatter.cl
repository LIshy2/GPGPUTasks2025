#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* positions,
    __global const uint* data,
    __global uint* result,
    unsigned int n,
    unsigned int b)
{
    int index = get_global_id(0);
    int all = n - positions[n - 1];
    if (index < n) {
        if (data[index] & (1 << b)) {
            // printf("put one bit %d=%d at %d\n", index, data[index], all + positions[index] - 1);
            result[all + positions[index] - 1] = data[index];
        } else {
            // printf("put zero bit %d=%d at %d\n", index, data[index], index - positions[index]);
            result[index - positions[index]] = data[index];
        }
    }
}
