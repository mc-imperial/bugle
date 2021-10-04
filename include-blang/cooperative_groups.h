#ifndef _COOPERATIVE_GROUPS_H_
#define _COOPERATIVE_GROUPS_H_

namespace cooperative_groups {
  class grid_group {
    public:
      __forceinline__ __device__ void sync() const {
          bugle_grid_barrier();
      }
  };
  
  class thread_block {
    public:
      __forceinline__ __device__ void sync() const {
          bugle_barrier(true, true);
      }
  };
  
  __forceinline__ __device__ grid_group this_grid();
  
  __forceinline__ __device__ thread_block this_thread_block();
  
  __forceinline__ __device__ void synchronize(grid_group g) {
      bugle_grid_barrier();
  }
  
  __forceinline__ __device__ void synchronize(thread_block t) {
      bugle_barrier(true, true);
  }
}

#endif