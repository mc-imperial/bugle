_CLC_OVERLOAD _CLC_INLINE __CLC_FLOAT __bugle_fast_length(__CLC_FLOATN p) {
  return __bugle_half_sqrt(dot(p, p));
}
