_CLC_DECL void __barrier(void);

_CLC_INLINE void barrier(cl_mem_fence_flags flags) {
  __barrier();
}
