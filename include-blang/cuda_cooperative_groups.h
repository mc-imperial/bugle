#ifndef CUDA_COOPERATIVE_GROUPS_H
#define CUDA_COOPERATIVE_GROUPS_H

namespace cooperative_groups {
  class grid_group {
    public:
      __inline__ __device__ void sync() const {
          bugle_grid_barrier();
      }
  };
  
  class thread_block {
    public:
      __inline__ __device__ void sync() const {
          bugle_barrier(true, true);
      }
  };
  
  __device__ grid_group this_grid();
  
  __device__ thread_block this_thread_block();
  
  static __inline__ __device__ void synchronize(grid_group g) {
      bugle_grid_barrier();
  }
  
  static __inline__ __device__ void synchronize(thread_block t) {
      bugle_barrier(true, true);
  }
}

#endif
