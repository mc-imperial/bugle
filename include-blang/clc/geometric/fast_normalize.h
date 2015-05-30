_CLC_OVERLOAD _CLC_INLINE float __bugle_fast_normalize(float p) {
  return sign(p);
}

_CLC_OVERLOAD int all(int2 p);

_CLC_OVERLOAD _CLC_INLINE float2 __bugle_fast_normalize(float2 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * half_rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(int3 p);

_CLC_OVERLOAD _CLC_INLINE float3 __bugle_fast_normalize(float3 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * half_rsqrt(dot(p, p));
}

_CLC_OVERLOAD int all(int4 p);

_CLC_OVERLOAD _CLC_INLINE float4 __bugle_fast_normalize(float4 p) {
  if (all(p == 0.0f))
    return p;
  else
    return p * half_rsqrt(dot(p, p));
}

#define fast_normalize __bugle_fast_normalize
