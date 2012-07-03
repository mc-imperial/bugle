#ifndef ANNOTATIONS_H
#define ANNOTATIONS_H

#ifdef __CUDA__
#define __DEVICE_QUALIFIER__ __device__
#else
#define __DEVICE_QUALIFIER__
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

/* Read set is empty */
__DEVICE_QUALIFIER__ bool __no_read(const char* array_name);

/* Read set is non-empty */
__DEVICE_QUALIFIER__ bool __read(const char* array_name);

/* Write set is empty */
__DEVICE_QUALIFIER__ bool __no_write(const char* array_name);

/* Write set is non-empty */
__DEVICE_QUALIFIER__ bool __write(const char* array_name);

/* Read offset */
__DEVICE_QUALIFIER__ int __read_offset(const char* array_name);

/* Write set is empty */
__DEVICE_QUALIFIER__ int __write_offset(const char* array_name);

/* If a read has occurred to 'array_name' then 'expr' must hold */
__DEVICE_QUALIFIER__ bool __read_implies(const char* array_name, bool expr);

/* If a write has occurred to 'array_name' then 'expr' must hold */
__DEVICE_QUALIFIER__ bool __write_implies(const char* array_name, bool expr);


#ifdef __OPENCL__
bool __points_to_global(const __global void* array, const char* array_name);
bool __points_to_local(const __local void* array, const char* array_name);
bool __points_to_private(const __private void* array, const char* array_name);
#endif

/* Inter-thread predicates */

__DEVICE_QUALIFIER__ int __other_int(int expr);
__DEVICE_QUALIFIER__ bool __other_bool(bool expr);

#define __uniform_int(X) ((X) == __other_int(X))
#define __uniform_bool(X) ((X) == __other_bool(X))

#define __distinct_int(X) ((X) != __other_int(X))
#define __distinct_bool(X) ((X) != __other_bool(X))

#define __all(X) ((X) & __other_bool(X))
#define __exclusive(X) (!(__all(X)))

/* Axioms */
#define __concatenate(x,y) x##y
#define __axiom_inner(x,y) __concatenate(x,y)
#define __axiom(expr) bool __axiom_inner(__axiom, __COUNTER__) () { return expr; }


/* Helpers */

#define __is_pow2(x) ((((x) & (x - 1)) == 0))
#define __mod_pow2(x,y) ((y - 1) & (x))

#endif
