/* atomic_cmpxchg(p, cmp, val), computes (old == cmp) ? val : old */

#define atomic_cmpxchg __bugle_atomic_cmpxchg
#define __CLC_FUNCTION bugle_atomic_cmpxchg

#define __CLC_DECLARE_ATOMIC_3_ARG(FUNCTION, ADDRSPACE, TYPE) \
    TYPE __##FUNCTION##_##ADDRSPACE##_##TYPE( \
           volatile ADDRSPACE TYPE *x, TYPE y, TYPE z); \
    _CLC_INLINE _CLC_OVERLOAD TYPE __##FUNCTION( \
           volatile ADDRSPACE TYPE *x, TYPE y, TYPE z) { \
        return __##FUNCTION##_##ADDRSPACE##_##TYPE(x, y, z); \
    }

#define __CLC_DECLARE_ATOMIC_ADDRSPACE_3_ARG(FUNCTION, TYPE) \
	__CLC_DECLARE_ATOMIC_3_ARG(FUNCTION, global, TYPE) \
	__CLC_DECLARE_ATOMIC_3_ARG(FUNCTION, local, TYPE)

__CLC_DECLARE_ATOMIC_ADDRSPACE_3_ARG(__CLC_FUNCTION, int)
__CLC_DECLARE_ATOMIC_ADDRSPACE_3_ARG(__CLC_FUNCTION, uint)

#undef __CLC_FUNCTION
#undef __CLC_DECLARE_ATOMIC_3_ARG
#undef __CLC_DECLARE_ATOMIC_ADDRESS_SPACE_3_ARG
