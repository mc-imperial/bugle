_CLC_OVERLOAD _CLC_INLINE float __bugle_normalize(float p) {
  return sign(p);
}

_CLC_OVERLOAD int all(int2 p);

_CLC_OVERLOAD _CLC_INLINE float2 __bugle_normalize(float2 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(int3 p);

_CLC_OVERLOAD _CLC_INLINE float3 __bugle_normalize(float3 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(int4 p);

_CLC_OVERLOAD _CLC_INLINE float4 __bugle_normalize(float4 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

#ifdef cl_khr_fp64

_CLC_OVERLOAD _CLC_INLINE double __bugle_normalize(double p) {
  return sign(p);
}

_CLC_OVERLOAD int all(long2 p);

_CLC_OVERLOAD _CLC_INLINE double2 __bugle_normalize(double2 p) {
  if (all(p == 0.0))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(long3 p);

_CLC_OVERLOAD _CLC_INLINE double3 __bugle_normalize(double3 p) {
  if (all(p == 0.0))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(long4 p);

_CLC_OVERLOAD _CLC_INLINE double4 __bugle_normalize(double4 p) {
  if (all(p == 0.0))
    return p;
  else
    return p * rsqrt(dot(p, p));
}

#endif

#define normalize __bugle_normalize
