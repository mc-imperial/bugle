_CLC_OVERLOAD _CLC_INLINE __CLC_FLOAT __bugle_length(__CLC_FLOATN p);

_CLC_OVERLOAD _CLC_INLINE __CLC_FLOAT __bugle_distance(
    __CLC_FLOATN p0, __CLC_FLOATN p1) {
  return __bugle_length(p0 - p1);
}
