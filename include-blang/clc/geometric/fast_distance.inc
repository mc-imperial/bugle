_CLC_OVERLOAD _CLC_DECL __CLC_FLOAT __bugle_fast_length(__CLC_FLOATN p);

_CLC_OVERLOAD _CLC_INLINE __CLC_FLOAT __bugle_fast_distance(
    __CLC_FLOATN p0, __CLC_FLOATN p1) {
  return __bugle_fast_length(p0 - p1);
}
