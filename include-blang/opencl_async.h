#ifndef OPENCL_ASYNC_H
#define OPENCL_ASYNC_H

#define event_t __bugle_event_t

typedef size_t event_t;

#define _ASYNC_WORK_GROUP_COPY_OVERLOAD(GENTYPE, DST_MEMORY_SPACE, SRC_MEMORY_SPACE) \
    event_t __async_work_group_copy_##SRC_MEMORY_SPACE##_to_##DST_MEMORY_SPACE##_##GENTYPE(DST_MEMORY_SPACE GENTYPE *dst, const SRC_MEMORY_SPACE GENTYPE *src, size_t num_elements, event_t event); \
    _CLC_INLINE _CLC_OVERLOAD event_t async_work_group_copy(DST_MEMORY_SPACE GENTYPE *dst, const SRC_MEMORY_SPACE GENTYPE *src, size_t num_elements, event_t event) { \
        return __async_work_group_copy_##SRC_MEMORY_SPACE##_to_##DST_MEMORY_SPACE##_##GENTYPE(dst, src, num_elements, event); \
    }

#define _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(GENTYPE) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD(GENTYPE, __global, __local) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD(GENTYPE, __local, __global) \

#define _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(SCALAR_GENTYPE) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(SCALAR_GENTYPE) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(SCALAR_GENTYPE##2) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(SCALAR_GENTYPE##4) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(SCALAR_GENTYPE##8) \
    _ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS(SCALAR_GENTYPE##16)
    
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(char)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(uchar)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(short)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(ushort)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(int)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(uint)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(long)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(ulong)
_ASYNC_WORK_GROUP_COPY_OVERLOAD_BOTH_DIRECTIONS_ALL_SIZES(float)
// Did not yet add double case, this should be there if cl_khr_fp64 is enabled

void wait_group_events(int num_events, event_t *event_list);

#endif
