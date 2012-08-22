#ifndef ANNOTATIONS_H
#define ANNOTATIONS_H

#ifdef __CUDA__
#define __DEVICE_QUALIFIER__ __device__
#else
#define __DEVICE_QUALIFIER__
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Loop invariant */
__DEVICE_QUALIFIER__ void __invariant(bool expr);

/* Function precondition */
__DEVICE_QUALIFIER__ void __requires(bool expr);

/* Function postcondition */
__DEVICE_QUALIFIER__ void __ensures(bool expr);

/* Return value of function, for use in postconditions */
__DEVICE_QUALIFIER__ int __return_val_int(void);
__DEVICE_QUALIFIER__ bool __return_val_bool(void);
#ifdef __OPENCL__
__DEVICE_QUALIFIER__ int4 __return_val_int4(void);
#endif

/* Old value of expression, for use in postconditions */
__DEVICE_QUALIFIER__ int __old_int(int);
__DEVICE_QUALIFIER__ bool __old_bool(bool);

/* Assumption */
__DEVICE_QUALIFIER__ void bugle_assume(bool expr);
#define __assume bugle_assume

/* Assertion */
__DEVICE_QUALIFIER__ void __assert(bool expr);
__DEVICE_QUALIFIER__ void __global_assert(bool expr);

/* Used to express whether a thread is enabled at a particuar point */
__DEVICE_QUALIFIER__ bool __enabled(void);

/* Maps to ==> */
__DEVICE_QUALIFIER__ bool __implies(bool expr1, bool expr2);

#define ptr_base_t int

#ifdef __OPENCL__

#define __POINTER_QUERY_OVERLOAD(NAME, MEMORY_SPACE, TYPE) \
    __DEVICE_QUALIFIER__ TYPE \
    __##NAME##_##MEMORY_SPACE \
    (const __##MEMORY_SPACE void* p); \
    __DEVICE_QUALIFIER__ __attribute__((overloadable)) \
    static __attribute__((always_inline)) TYPE \
    __##NAME(const __##MEMORY_SPACE void* p) { \
      return __##NAME##_##MEMORY_SPACE(p); \
    }

#define __POINTER_QUERY(NAME, TYPE) \
    __POINTER_QUERY_OVERLOAD(NAME, local, TYPE) \
    __POINTER_QUERY_OVERLOAD(NAME, global, TYPE) \
    
/* Read/write set is non-empty */
__POINTER_QUERY(read, bool)
__POINTER_QUERY(write, bool)

/* Read/write set is empty */
#define __no_read(p) !__read(p)
#define __no_write(p) !__write(p)

/* Read/write offset */
__POINTER_QUERY(read_offset, int)
__POINTER_QUERY(write_offset, int)

/* Pointer base */
__POINTER_QUERY(ptr_base, ptr_base_t);

/* Pointer offset */
__POINTER_QUERY(ptr_offset, int);

#endif

/* Inter-thread predicates */

__DEVICE_QUALIFIER__ int __other_int(int expr);
__DEVICE_QUALIFIER__ bool __other_bool(bool expr);
__DEVICE_QUALIFIER__ ptr_base_t __other_ptr_base(ptr_base_t expr);

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
#define __axiom(expr) extern "C" __DEVICE_QUALIFIER__ bool __axiom_inner(__axiom, __COUNTER__) () { return expr; }
#else
#define __axiom(expr) bool __axiom_inner(__axiom, __COUNTER__) () { return expr; }
#endif


/* Helpers */

#define __is_pow2(x) ((((x) & (x - 1)) == 0))
#define __mod_pow2(x,y) ((y - 1) & (x))

#ifdef __cplusplus
}
#endif

#ifdef __CUDA__
// TODO: remove these functions once the CUDA benchmarks have been moved to the
// new builtins.

/* Read set is empty */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __no_read(const char* array_name) { return false; }

/* Read set is non-empty */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __read(const char* array_name) { return false; }

/* Write set is empty */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __no_write(const char* array_name) { return false; }

/* Write set is non-empty */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __write(const char* array_name) { return false; }

/* Read offset */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ int __read_offset(const char* array_name) { return 0; }

/* Write set is empty */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ int __write_offset(const char* array_name) { return 0; }

/* If a read has occurred to 'array_name' then 'expr' must hold */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __read_implies(const char* array_name, bool expr) { return false; }

/* If a write has occurred to 'array_name' then 'expr' must hold */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __write_implies(const char* array_name, bool expr) { return false; }

/* 'expr' may hold for at most one thread */
__attribute__((always_inline)) __DEVICE_QUALIFIER__ bool __at_most_one(bool expr) { return false; }

#endif

#endif
