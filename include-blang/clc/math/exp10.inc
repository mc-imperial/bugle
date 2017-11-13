_CLC_OVERLOAD _CLC_INLINE __CLC_GENTYPE __bugle_exp10(__CLC_GENTYPE val) {
  // exp10(x) = exp2(x * log2(10))
#if __CLC_FPSIZE == 32
  #define __CLC_M_LOG210_F 0x1.a934f0p+1f
  return __bugle_exp_internal_exp2(val * __CLC_M_LOG210_F);
  #undef __CLC_M_LOG210_F
#elif __CLC_FPSIZE == 64
  #define __CLC_M_LOG210 0x1.a934f0979a371p+1
  return __bugle_exp_internal_exp2(val * __CLC_M_LOG210);
  #undef __CLC_M_LOG210
#else
#error unknown _CLC_FPSIZE
#endif
}
