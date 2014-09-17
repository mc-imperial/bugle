#define atomic_xchg __bugle_atomic_xchg
#define __CLC_FUNCTION bugle_atomic_xchg
#define __CLC_NEED_ATOMIC_FLOAT
#include <clc/atomic/atomic_decl.inc>
#undef __CLC_FUNCTION
