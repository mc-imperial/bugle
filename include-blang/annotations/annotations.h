#ifndef ANNOTATIONS_H
#define ANNOTATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Turning temporal nature of loads on and off */
_DEVICE_QUALIFIER void __non_temporal_loads_begin(void);
_DEVICE_QUALIFIER void __non_temporal_loads_end(void);
    
/* Loop invariant */
_DEVICE_QUALIFIER void __invariant(bool expr);

#define __invariant(X) \
    __non_temporal_loads_begin(), \
    __invariant(X), \
    __non_temporal_loads_end()
    
/* Function precondition */
_DEVICE_QUALIFIER void __requires(bool expr);
_DEVICE_QUALIFIER void __global_requires(bool expr);

/* Function postcondition */
_DEVICE_QUALIFIER void __ensures(bool expr);
_DEVICE_QUALIFIER void __global_ensures(bool expr);

/* Return value of function, for use in postconditions */
_DEVICE_QUALIFIER int __return_val_int(void);
_DEVICE_QUALIFIER bool __return_val_bool(void);
_DEVICE_QUALIFIER void* __return_val_ptr(void);
#ifdef __OPENCL_VERSION__
_DEVICE_QUALIFIER int4 __return_val_int4(void);
#endif

/* Old value of expression, for use in postconditions */
_DEVICE_QUALIFIER int __old_int(int);
_DEVICE_QUALIFIER bool __old_bool(bool);

/* Assumption */
#define __assume(e) bugle_assume(e)

/* Assertion */
_DEVICE_QUALIFIER void __assert(bool expr);
_DEVICE_QUALIFIER void __global_assert(bool expr);

/* Used to express whether a thread is enabled at a particuar point */
_DEVICE_QUALIFIER bool __enabled(void);

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
_FUNCTION_FROM_POINTER_TO_TYPE(read_offset, int)
_FUNCTION_FROM_POINTER_TO_TYPE(write_offset, int)

/* Pointer base */
_FUNCTION_FROM_POINTER_TO_TYPE(ptr_base, ptr_base_t);

/* Pointer offset */
_FUNCTION_FROM_POINTER_TO_TYPE(ptr_offset, int);
    
/* Read/write set is empty */
#define __no_read(p) !__read(p)
#define __no_write(p) !__write(p)

#define __read_implies(p, e) __implies(__read(p), e)
#define __write_implies(p, e) __implies(__write(p), e)

/* Used in specifications to say how a pointer is accessed */
_FUNCTION_FROM_POINTER_TO_VOID(reads_from);
_FUNCTION_FROM_POINTER_TO_VOID(writes_to);
    
#ifdef __OPENCL_VERSION__
void __array_snapshot_local(__local void* dst, __local void* src);
void __array_snapshot_global(__global void* dst, __global void* src);

_CLC_OVERLOAD _CLC_INLINE void __array_snapshot(__local void* dst, __local void* src) {
    return __array_snapshot_local(dst, src);
}

_CLC_OVERLOAD _CLC_INLINE void __array_snapshot(__global void* dst, __global void* src) {
    return __array_snapshot_global(dst, src);
}
#endif

#ifdef __CUDA_ARCH__
void __array_snapshot(void* dst, void* src);
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

#define __same_group (get_group_id(0) == __other_int(get_group_id(0)) \
                    & get_group_id(1) == __other_int(get_group_id(1)) \
                    & get_group_id(2) == __other_int(get_group_id(2)))

/* Axioms */
#define __concatenate(x,y) x##y
#define __axiom_inner(x,y) __concatenate(x,y)

#ifdef __cplusplus
#define __axiom(expr) extern "C" _DEVICE_QUALIFIER bool __axiom_inner(__axiom, __COUNTER__) () { return expr; }
#else
#define __axiom(expr) bool __axiom_inner(__axiom, __COUNTER__) () { return expr; }
#endif

/* Barrier invariants */

#include "barrier_invariants.h"
    
/* Helpers */

#define __is_pow2(x) ((((x) & (x - 1)) == 0))
#define __mod_pow2(x,y) ((y - 1) & (x))

unsigned __add_noovfl_unsigned(unsigned x, unsigned y);
int __add_noovfl_signed(int x, int y);

int __ite(bool b, int x, int y);

#define ADD_ABSTRACT(TYPE) \
    unsigned TYPE __add_abstract_##TYPE(unsigned TYPE x, unsigned TYPE y); \
    unsigned TYPE __add_abstract_1_##TYPE(unsigned TYPE x, unsigned TYPE y); \
    unsigned TYPE __add_abstract_2_##TYPE(unsigned TYPE x, unsigned TYPE y); \
    unsigned TYPE __add_abstract_3_##TYPE(unsigned TYPE x, unsigned TYPE y); \
    static __attribute__((always_inline)) __attribute__((overloadable)) unsigned TYPE __add_abstract(unsigned TYPE x, unsigned TYPE y) { \
      return __add_abstract_##TYPE(x, y); \
    } \
    static __attribute__((always_inline)) __attribute__((overloadable)) unsigned TYPE __add_abstract_1(unsigned TYPE x, unsigned TYPE y) { \
      return __add_abstract_1_##TYPE(x, y); \
    } \
    static __attribute__((always_inline)) __attribute__((overloadable)) unsigned TYPE __add_abstract_2(unsigned TYPE x, unsigned TYPE y) { \
      return __add_abstract_2_##TYPE(x, y); \
    } \
    static __attribute__((always_inline)) __attribute__((overloadable)) unsigned TYPE __add_abstract_3(unsigned TYPE x, unsigned TYPE y) { \
      return __add_abstract_3_##TYPE(x, y); \
    }

ADD_ABSTRACT(char)
ADD_ABSTRACT(short)
ADD_ABSTRACT(int)
ADD_ABSTRACT(long)

#ifdef __cplusplus
}
#endif

#endif
