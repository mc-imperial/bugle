_CLC_INLINE void barrier(cl_mem_fence_flags flags) {
  bugle_barrier(flags & CLK_LOCAL_MEM_FENCE, flags & CLK_GLOBAL_MEM_FENCE);
}
