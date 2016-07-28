#ifndef ANNOTATIONS_H
#define ANNOTATIONS_H

#ifdef NO_ANNOTATIONS_H
#error no_annotations.h must be included after annotations.h
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __OPENCL_VERSION__
#define _BUGLE_INLINE _CLC_INLINE
#endif

#ifdef __CUDA_ARCH__
#define _BUGLE_INLINE static __attribute__((always_inline))
#endif


/* Turning temporal nature of loads on and off */
_DEVICE_QUALIFIER void __non_temporal_loads_begin(void);
_DEVICE_QUALIFIER void __non_temporal_loads_end(void);

/* Loop invariant */
_DEVICE_QUALIFIER void __invariant(bool expr);
_DEVICE_QUALIFIER void __global_invariant(bool expr);
_DEVICE_QUALIFIER void __candidate_invariant(bool expr);
_DEVICE_QUALIFIER void __candidate_global_invariant(bool expr);

/* Function-wide invariant: will automatically become a global
   invariant for every loop in the function.  The candidate
   version becomes an independent candidate invariant for each
   loop in the function.
*/
_DEVICE_QUALIFIER void __function_wide_invariant(bool expr);
_DEVICE_QUALIFIER void __function_wide_candidate_invariant(bool expr);

#define __invariant(X) \
    __non_temporal_loads_begin(), \
    __invariant(X), \
    __non_temporal_loads_end()

#define __global_invariant(X) \
    __non_temporal_loads_begin(), \
    __global_invariant(X), \
    __non_temporal_loads_end()

#define __candidate_invariant(X) \
    __non_temporal_loads_begin(), \
    __candidate_invariant(X), \
    __non_temporal_loads_end()

#define __candidate_global_invariant(X) \
    __non_temporal_loads_begin(), \
    __candidate_global_invariant(X), \
    __non_temporal_loads_end()


/* Function precondition */
_DEVICE_QUALIFIER void __requires(bool expr);
_DEVICE_QUALIFIER void __global_requires(bool expr);

#define __requires(X) \
    __non_temporal_loads_begin(), \
    __requires(X), \
    __non_temporal_loads_end()

#define __global_requires(X) \
    __non_temporal_loads_begin(), \
    __global_requires(X), \
    __non_temporal_loads_end()

/* Function postcondition */
_DEVICE_QUALIFIER void __ensures(bool expr);
_DEVICE_QUALIFIER void __global_ensures(bool expr);

#define __ensures(X) \
    __non_temporal_loads_begin(), \
    __ensures(X), \
    __non_temporal_loads_end()

#define __global_ensures(X) \
    __non_temporal_loads_begin(), \
    __global_ensures(X), \
    __non_temporal_loads_end()

/* Return value of function, for use in postconditions */
_DEVICE_QUALIFIER int __return_val_int(void);
_DEVICE_QUALIFIER bool __return_val_bool(void);
_DEVICE_QUALIFIER void* __return_val_ptr(void);
_DEVICE_QUALIFIER void* __return_val_funptr(void);
#define __return_val_funptr(X) \
  ((X)__return_val_funptr())
#ifdef __OPENCL_VERSION__
_DEVICE_QUALIFIER int4 __return_val_int4(void);
#endif

/* Old value of expression, for use in postconditions */
_DEVICE_QUALIFIER int __old_int(int);
_DEVICE_QUALIFIER bool __old_bool(bool);

/* Assumption */
#define __assume(X) \
    __non_temporal_loads_begin(), \
    bugle_assume(X), \
    __non_temporal_loads_end()

/* Assertion */
_DEVICE_QUALIFIER void __assert(bool expr);
_DEVICE_QUALIFIER void __global_assert(bool expr);
_DEVICE_QUALIFIER void __candidate_assert(bool expr);
_DEVICE_QUALIFIER void __candidate_global_assert(bool expr);

#define __unsafe_assert(X) \
    __non_temporal_loads_begin(), \
    __assert(X), \
    __non_temporal_loads_end()

#define __unsafe_global_assert(X) \
    __non_temporal_loads_begin(), \
    __global_assert(X), \
    __non_temporal_loads_end()

#define __unsafe_candidate_assert(X) \
    __non_temporal_loads_begin(), \
    __candidate_assert(X), \
    __non_temporal_loads_end()

#define __unsafe_candidate_global_assert(X) \
    __non_temporal_loads_begin(), \
    __candidate_global_assert(X), \
    __non_temporal_loads_end()

/* Used to express whether a thread is enabled at a particuar point */
_DEVICE_QUALIFIER bool __enabled(void);
/* Used to express whether a thread is enabled in the block immediately
   dominating a particuar point */
_DEVICE_QUALIFIER bool __dominator_enabled(void);

/* Maps to ==> */
_DEVICE_QUALIFIER bool __implies(bool expr1, bool expr2);

#define ptr_base_t int

#ifdef __OPENCL_VERSION__
#include <annotations/cl_pointer_annotations.h>
#endif

#ifdef __CUDA_ARCH__
#include <annotations/cu_pointer_annotations.h>
#endif

/* Read/write set is non-empty */
_FUNCTION_FROM_POINTER_TO_TYPE(read, bool)
_FUNCTION_FROM_POINTER_TO_TYPE(write, bool)

/* Read/write offset */
_FUNCTION_FROM_POINTER_TO_TYPE(read_offset_bytes, size_t)
_FUNCTION_FROM_POINTER_TO_TYPE(write_offset_bytes, size_t)

/* Pointer base */
_FUNCTION_FROM_POINTER_TO_TYPE(ptr_base, ptr_base_t)

/* Pointer offset */
_FUNCTION_FROM_POINTER_TO_TYPE(ptr_offset_bytes, size_t)

/* Read/write set is empty */
#define __no_read(p) !__read(p)
#define __no_write(p) !__write(p)

#define __read_implies(p, e) __implies(__read(p), e)
#define __write_implies(p, e) __implies(__write(p), e)

/* Used in specifications to say how a pointer is accessed */
_FUNCTION_FROM_POINTER_TO_VOID(reads_from)
_FUNCTION_FROM_POINTER_TO_VOID(writes_to)


#ifdef __OPENCL_VERSION__
bool __atomic_has_taken_value_local(__local unsigned *atomic_array, size_t offset, unsigned value);
bool __atomic_has_taken_value_global(__global unsigned *atomic_array, size_t offset, unsigned value);

_CLC_OVERLOAD _CLC_INLINE bool __atomic_has_taken_value(__local unsigned *atomic_array, size_t offset, unsigned value) {
    return __atomic_has_taken_value_local(atomic_array, offset, value);
}

_CLC_OVERLOAD _CLC_INLINE bool __atomic_has_taken_value(__global unsigned *atomic_array, size_t offset, unsigned value) {
    return __atomic_has_taken_value_global(atomic_array, offset, value);
}
#endif

#ifdef __CUDA_ARCH__
__device__ bool __atomic_has_taken_value(unsigned *atomic_array, unsigned offset, unsigned value);
#endif

#ifdef __OPENCL_VERSION__
void __array_snapshot_local(__local void* dst, __local void* src);
void __array_snapshot_global(__global void* dst, __global void* src);

_CLC_OVERLOAD _CLC_INLINE void __array_snapshot(__local void* dst, __local void* src) {
    __array_snapshot_local(dst, src);
}

_CLC_OVERLOAD _CLC_INLINE void __array_snapshot(__global void* dst, __global void* src) {
    __array_snapshot_global(dst, src);
}
#endif

#ifdef __CUDA_ARCH__
__device__ void __array_snapshot(void* dst, void* src);
#endif

/* Inter-thread predicates */

_DEVICE_QUALIFIER int __other_int(int expr);
_DEVICE_QUALIFIER bool __other_bool(bool expr);

#define __uniform_int(X) ((X) == __other_int(X))
#define __uniform_bool(X) ((X) == __other_bool(X))
#define __uniform_ptr_base(X) ((X) == __other_ptr_base(X))

#define __distinct_int(X) ((X) != __other_int(X))
#define __distinct_bool(X) ((X) != __other_bool(X))
#define __distinct_ptr_base(X) ((X) != __other_ptr_base(X))

#define __all(X) ((X) & __other_bool(X))
#define __exclusive(X) (!(__all(X)))

#ifdef __OPENCL_VERSION__
#define __same_group (get_group_id(0) == __other_int(get_group_id(0)) \
                    & get_group_id(1) == __other_int(get_group_id(1)) \
                    & get_group_id(2) == __other_int(get_group_id(2)))
#endif

#ifdef __CUDA_ARCH__
#define __same_group (blockIdx.x == __other_int(blockIdx.x) \
                    & blockIdx.y == __other_int(blockIdx.y) \
                    & blockIdx.z == __other_int(blockIdx.z))
#endif

/* Axioms */
#define __concatenate(x,y) x##y
#define __axiom_inner(x,y) __concatenate(x,y)

#ifdef __cplusplus
#define __axiom_middle(expr, counter) \
  extern "C" _DEVICE_QUALIFIER bool __axiom_inner(__axiom, counter) (void); \
  extern "C" _DEVICE_QUALIFIER bool __axiom_inner(__axiom, counter) (void) \
  { return expr; }
#define __axiom(expr) __axiom_middle(expr,__COUNTER__)
#else
#define __axiom_middle(expr, counter) \
  bool __axiom_inner(__axiom, counter) (void); \
  bool __axiom_inner(__axiom, counter) (void) \
  { return expr; }
#define __axiom(expr) __axiom_middle(expr, __COUNTER__)
#endif

/* Barrier invariants */

#include "barrier_invariants.h"

/* Helpers */

#define __is_pow2(x) ((((x) & (x - 1)) == 0))
#define __mod_pow2(x,y) ((y - 1) & (x))

/* Non overflowing addition */

#define NOOVFL_DECL(TYPE) \
    _DEVICE_QUALIFIER TYPE __add_noovfl_##TYPE(TYPE x, TYPE y); \
    _DEVICE_QUALIFIER unsigned TYPE __add_noovfl_unsigned_##TYPE(unsigned TYPE x, unsigned TYPE y);

NOOVFL_DECL(char)
NOOVFL_DECL(short)
NOOVFL_DECL(int)
NOOVFL_DECL(long)

/* Non overflowing addition predicates */

#include "noovfl_predicates_autogenerated_definitions.h"

/* If-Then-Else */

#define ITE_DECL(TYPE) \
    _DEVICE_QUALIFIER TYPE __ite_##TYPE(bool b, TYPE x, TYPE y); \
    _DEVICE_QUALIFIER unsigned TYPE __ite_unsigned_##TYPE(bool b, unsigned TYPE x, unsigned TYPE y); \
    _DEVICE_QUALIFIER _BUGLE_INLINE __attribute__((overloadable)) TYPE __ite(bool b, TYPE x, TYPE y) { \
      return __ite_##TYPE(b, x, y); \
    } \
    _DEVICE_QUALIFIER _BUGLE_INLINE __attribute__((overloadable)) unsigned TYPE __ite(bool b, unsigned TYPE x, unsigned TYPE y) { \
      return __ite_unsigned_##TYPE(b, x, y); \
    }

ITE_DECL(char)
ITE_DECL(short)
ITE_DECL(int)
ITE_DECL(long)

#undef ITE_DECL

#if __BUGLE_64__ && !__OPENCL_VERSION__
_DEVICE_QUALIFIER size_t __ite_size_t(bool b, size_t x, size_t y);
_DEVICE_QUALIFIER _BUGLE_INLINE __attribute__((overloadable)) size_t __ite(bool b, size_t x, size_t y) {
  return __ite_size_t(b, x, y);
}
#endif

/* Addition */

#define ADD_DECL(TYPE) \
    _DEVICE_QUALIFIER TYPE __add_##TYPE(TYPE x, TYPE y); \
    _DEVICE_QUALIFIER unsigned TYPE __add_unsigned_##TYPE(unsigned TYPE x, unsigned TYPE y); \
    _DEVICE_QUALIFIER _BUGLE_INLINE __attribute__((overloadable)) TYPE __add(TYPE x, TYPE y) { \
      return __add_##TYPE(x, y); \
    } \
    _DEVICE_QUALIFIER _BUGLE_INLINE __attribute__((overloadable)) unsigned TYPE __add(unsigned TYPE x, unsigned TYPE y) { \
      return __add_unsigned_##TYPE(x, y); \
    }

ADD_DECL(char)
ADD_DECL(short)
ADD_DECL(int)
ADD_DECL(long)

#undef ADD_DECL

/* Uninterpreted functions */

#define DECLARE_UF_BINARY(NAME, ARG1TYPE, ARG2TYPE, RETURNTYPE) \
    _DEVICE_QUALIFIER RETURNTYPE \
    __uninterpreted_function_##NAME(ARG1TYPE, ARG2TYPE); \
    _DEVICE_QUALIFIER _BUGLE_INLINE RETURNTYPE \
    NAME(ARG1TYPE x, ARG2TYPE y) { \
       return __uninterpreted_function_##NAME(x, y);   \
    }

#define DECLARE_UF_UNARY(NAME, ARG1TYPE, RETURNTYPE) \
    _DEVICE_QUALIFIER RETURNTYPE \
    __uninterpreted_function_##NAME(ARG1TYPE); \
    _DEVICE_QUALIFIER _BUGLE_INLINE RETURNTYPE \
    NAME(ARG1TYPE x) { \
       return __uninterpreted_function_##NAME(x);   \
    }

#ifdef __cplusplus
}
#endif

#endif
