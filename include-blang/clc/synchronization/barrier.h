#define barrier(flags) \
  bugle_barrier((flags) & CLK_LOCAL_MEM_FENCE, (flags) & CLK_GLOBAL_MEM_FENCE)
