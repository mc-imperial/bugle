#define __CLC_DECLARE_ATOM_INNER(FUNCTION, TYPE) \
    TYPE __bugle_atomic_##FUNCTION##_global_##TYPE( \
           volatile global TYPE *x); \
    _CLC_INLINE _CLC_OVERLOAD TYPE __bugle_atom_##FUNCTION( \
           global TYPE *x) { \
        return __bugle_atomic_##FUNCTION##_global_##TYPE(x); \
    }

#define __CLC_DECLARE_ATOM(FUNCTION, TYPE) \
    __CLC_DECLARE_ATOM_INNER(FUNCTION, TYPE)

__CLC_DECLARE_ATOM(__CLC_FUNCTION, int)
__CLC_DECLARE_ATOM(__CLC_FUNCTION, uint)

#undef __CLC_DECLARE_ATOM_INNER
#undef __CLC_DECLARE_ATOM
