#ifndef OPENCL_BUILTINS_H
#define OPENCL_BUILTINS_H

// 6.12.2: Math functions

#define _MATH_UNARY_FUNC_OVERLOAD(NAME, GENTYPE) \
    GENTYPE __##NAME##_##GENTYPE(GENTYPE x); \
    _CLC_INLINE _CLC_OVERLOAD GENTYPE NAME(GENTYPE x) { \
        return __##NAME##_##GENTYPE(x); \
    }

#define _MATH_BINARY_FUNC_OVERLOAD(NAME, GENTYPE) \
    GENTYPE __##NAME##_##GENTYPE(GENTYPE x, GENTYPE y); \
    _CLC_INLINE _CLC_OVERLOAD GENTYPE NAME(GENTYPE x, GENTYPE y) { \
        return __##NAME##_##GENTYPE(x, y); \
    }

#define _MATH_TERNARY_FUNC_OVERLOAD(NAME, GENTYPE) \
    GENTYPE __##NAME##_##GENTYPE(GENTYPE x, GENTYPE y, GENTYPE z); \
    _CLC_INLINE _CLC_OVERLOAD GENTYPE NAME(GENTYPE x, GENTYPE y, GENTYPE z) { \
        return __##NAME##_##GENTYPE(x, y, z); \
    }


#define _FLOAT_UNARY_MACRO(NAME)          \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float) \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float2) \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float3) \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float4) \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float8) \
    _MATH_UNARY_FUNC_OVERLOAD(NAME, float16)

#define _FLOAT_BINARY_MACRO(NAME)          \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float) \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float2) \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float3) \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float4) \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float8) \
    _MATH_BINARY_FUNC_OVERLOAD(NAME, float16)

#define _FLOAT_TERNARY_MACRO(NAME)          \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float) \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float2) \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float3) \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float4) \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float8) \
    _MATH_TERNARY_FUNC_OVERLOAD(NAME, float16)

// Table 6.8

_FLOAT_UNARY_MACRO(cbrt)
_FLOAT_UNARY_MACRO(logb)

// Table 6.9

_FLOAT_UNARY_MACRO(half_cos)
_FLOAT_UNARY_MACRO(half_divide)
_FLOAT_UNARY_MACRO(half_exp)
_FLOAT_UNARY_MACRO(half_exp2)
_FLOAT_UNARY_MACRO(half_exp10)
_FLOAT_UNARY_MACRO(half_log)
_FLOAT_UNARY_MACRO(half_log2)
_FLOAT_UNARY_MACRO(half_log10)
_FLOAT_BINARY_MACRO(half_powr)
_FLOAT_UNARY_MACRO(half_recip)
_FLOAT_UNARY_MACRO(half_rsqrt)
_FLOAT_UNARY_MACRO(half_sin)
_FLOAT_UNARY_MACRO(half_sqrt)
_FLOAT_UNARY_MACRO(half_tan)

#define NAN      (0.0f/0.0f)

// 6.12.5: Geometric functions

_CLC_INLINE _CLC_OVERLOAD float fast_length(float p) {
    return half_sqrt(p*p);
}
_CLC_INLINE _CLC_OVERLOAD float fast_length(float2 p) {
    return half_sqrt(p.x*p.x + p.y*p.y);
}
_CLC_INLINE _CLC_OVERLOAD float fast_length(float3 p) {
    return half_sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}
_CLC_INLINE _CLC_OVERLOAD float fast_length(float4 p) {
    return half_sqrt(p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w);
}



_CLC_INLINE _CLC_OVERLOAD float fast_normalize(float p) {
    return p*half_rsqrt(p*p);
}
_CLC_INLINE _CLC_OVERLOAD float2 fast_normalize(float2 p) {
    return p*half_rsqrt(p.x*p.x + p.y*p.y);
}
_CLC_INLINE _CLC_OVERLOAD float3 fast_normalize(float3 p) {
    return p*half_rsqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}
_CLC_INLINE _CLC_OVERLOAD float4 fast_normalize(float4 p) {
    return p*half_rsqrt(p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w);
}

#endif

