#define __CLC_FUNCTION __bugle_exp10_internal_exp2
#define __CLC_INTRINSIC "llvm.exp2"
#include <clc/math/unary_intrin.inc>

#define __CLC_BODY <clc/math/exp10.inc>
#include <clc/math/gentype.inc>
#define exp10 __bugle_exp10
