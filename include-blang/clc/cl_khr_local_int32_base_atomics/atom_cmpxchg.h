/* atomic_cmpxchg(p, cmp, val), computes (old == cmp) ? val : old */

#define atom_cmpxchg __bugle_atom_cmpxchg
#define __CLC_FUNCTION cmpxchg

#define __CLC_DECLARE_ATOM_3_ARG_INNER(FUNCTION, TYPE) \
    TYPE __bugle_atomic_##FUNCTION##_local_##TYPE( \
           volatile local TYPE *x, TYPE y, TYPE z); \
    _CLC_INLINE _CLC_OVERLOAD TYPE __bugle_atom_##FUNCTION( \
           local TYPE *x, TYPE y, TYPE z) { \
        return __bugle_atomic_##FUNCTION##_local_##TYPE(x, y, z); \
    }

#define __CLC_DECLARE_ATOM_3_ARG(FUNCTION, TYPE) \
	__CLC_DECLARE_ATOM_3_ARG_INNER(FUNCTION, TYPE)

__CLC_DECLARE_ATOM_3_ARG(__CLC_FUNCTION, int)
__CLC_DECLARE_ATOM_3_ARG(__CLC_FUNCTION, uint)

#undef __CLC_FUNCTION
#undef __CLC_DECLARE_ATOM_3_ARG_INNER
#undef __CLC_DECLARE_ATOM_3_ARG
