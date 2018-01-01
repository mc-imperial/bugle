#ifndef CUDA_VECTORS_H
#define CUDA_VECTORS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++11-extensions"

typedef unsigned short ushort;
typedef unsigned int uint;

/* See Table B-1 in CUDA Specification */
/* From vector_functions.h */

#define __MAKE_VECTOR_OPERATIONS(TYPE,NAME) \
  typedef struct {   \
    TYPE x;          \
  } NAME##1;         \
  typedef struct {   \
    TYPE x, y;       \
  } NAME##2;         \
  typedef struct {   \
    TYPE x, y, z;    \
  } NAME##3;         \
  typedef struct {   \
    TYPE x, y, z, w; \
  } NAME##4;         \
  __host__ __device__ static __inline__ NAME##1 make_##NAME##1(TYPE x) \
  { \
    return { x }; \
  } \
  __host__ __device__ static __inline__ NAME##2 make_##NAME##2(TYPE x, TYPE y) \
  { \
    return { x, y }; \
  } \
  __host__ __device__ static __inline__ NAME##2 make_##NAME##2(TYPE x) \
  { \
    return make_##NAME##2(x, x); \
  } \
  __host__ __device__ static __inline__ NAME##3 make_##NAME##3(TYPE x, TYPE y, TYPE z) \
  { \
    return { x, y, z }; \
  } \
  __host__ __device__ static __inline__ NAME##3 make_##NAME##3(TYPE x) \
  { \
    return make_##NAME##3(x, x, x); \
  } \
  __host__ __device__ static __inline__ NAME##4 make_##NAME##4(TYPE x, TYPE y, TYPE z, TYPE w) \
  { \
    return { x, y, z, w }; \
  } \
  __device__ static __inline__ NAME##4 make_##NAME##4(TYPE x) \
  { \
    return make_##NAME##4(x, x, x, x); \
  }

__MAKE_VECTOR_OPERATIONS(signed char,char)
__MAKE_VECTOR_OPERATIONS(unsigned char,uchar)
__MAKE_VECTOR_OPERATIONS(short, short)
__MAKE_VECTOR_OPERATIONS(unsigned short,ushort)
__MAKE_VECTOR_OPERATIONS(int,int)
__MAKE_VECTOR_OPERATIONS(unsigned int,uint)
__MAKE_VECTOR_OPERATIONS(long,long)
__MAKE_VECTOR_OPERATIONS(unsigned long,ulong)
__MAKE_VECTOR_OPERATIONS(long long,longlong)
__MAKE_VECTOR_OPERATIONS(unsigned long long,ulonglong)
__MAKE_VECTOR_OPERATIONS(float,float)
__MAKE_VECTOR_OPERATIONS(double,double)

#undef __MAKE_VECTOR_OPERATIONS

/* from helper_math.h */

#define __MAKE_VECTOR_FROM_SIMILAR(TYPE,ZERO) \
  __host__ __device__ static __inline__ TYPE##2 make_##TYPE##2(TYPE##3 a) \
  { \
    return make_##TYPE##2(a.x, a.y);  /* discards z */ \
  } \
  __host__ __device__ static __inline__ TYPE##3 make_##TYPE##3(TYPE##2 a) \
  { \
    return make_##TYPE##3(a.x, a.y, ZERO); \
  } \
  __host__ __device__ static __inline__ TYPE##3 make_##TYPE##3(TYPE##2 a, TYPE s) \
  { \
    return make_##TYPE##3(a.x, a.y, s); \
  } \
  __host__ __device__ static __inline__ TYPE##4 make_##TYPE##4(TYPE##3 a) \
  { \
    return make_##TYPE##4(a.x, a.y, a.z, ZERO); \
  } \
  __host__ __device__ static __inline__ TYPE##4 make_##TYPE##4(TYPE##3 a, TYPE w) \
  { \
    return make_##TYPE##4(a.x, a.y, a.z, w); \
  }

__MAKE_VECTOR_FROM_SIMILAR(float,0.0f)
__MAKE_VECTOR_FROM_SIMILAR(int,0)
__MAKE_VECTOR_FROM_SIMILAR(uint,0)

#undef __MAKE_VECTOR_FROM_SIMILAR

#define __MAKE_CONVERT(TYPE_TO,TYPE_FROM) \
  __host__ __device__ static __inline__ TYPE_TO##2 make_##TYPE_TO##2(TYPE_FROM##2 a) \
  { \
    return make_##TYPE_TO##2(TYPE_TO(a.x), TYPE_TO(a.y)); \
  } \
  __host__ __device__ static __inline__ TYPE_TO##3 make_##TYPE_TO##3(TYPE_FROM##3 a) \
  { \
    return make_##TYPE_TO##3(TYPE_TO(a.x), TYPE_TO(a.y), TYPE_TO(a.z)); \
  } \
  __host__ __device__ static __inline__ TYPE_TO##4 make_##TYPE_TO##4(TYPE_FROM##4 a) \
  { \
    return make_##TYPE_TO##4(TYPE_TO(a.x), TYPE_TO(a.y), TYPE_TO(a.z), TYPE_TO(a.w)); \
  }

__MAKE_CONVERT(float,int)
__MAKE_CONVERT(float,uint)
__MAKE_CONVERT(int,uint)
__MAKE_CONVERT(int,float)
__MAKE_CONVERT(uint,int)

#undef __MAKE_CONVERT_TWO

__host__ __device__ static __inline__ float3 make_float3(float4 a)
{
  return make_float3(a.x, a.y, a.z);  /* discards w */
}

__host__ __device__ static __inline__ uint3 make_uint3(uint4 a)
{
  return make_uint3(a.x, a.y, a.z);  /* discards w */
}

#define __MAKE_NEGATIVE(TYPE) \
  __device__ static __inline__ TYPE##2 operator-(TYPE##2 &a) \
  { \
    return make_##TYPE##2(-a.x, -a.y); \
  } \
  __device__ static __inline__ TYPE##3 operator-(TYPE##3 &a) \
  { \
    return make_##TYPE##3(-a.x, -a.y, -a.z); \
  } \
  __device__ static __inline__ TYPE##4 operator-(TYPE##4 &a) \
  { \
    return make_##TYPE##4(-a.x, -a.y, -a.z, -a.w); \
  }

__MAKE_NEGATIVE(float)
__MAKE_NEGATIVE(int)

#undef __MAKE_NEGATIVE

#define __MAKE_PLUS(TYPE) \
  __device__ static __inline__ TYPE##2 operator+(TYPE##2 a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a.x + b.x, a.y + b.y); \
  } \
  __device__ static __inline__ void operator+=(TYPE##2 &a, TYPE##2 b) \
  { \
    a = make_##TYPE##2(a.x + b.x, a.y + b.y); \
  } \
  __device__ static __inline__ TYPE##2 operator+(TYPE##2 a, TYPE b) \
  { \
    return make_##TYPE##2(a.x + b, a.y + b); \
  } \
  __device__ static __inline__ TYPE##2 operator+(TYPE a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a + b.x, a + b.y); \
  } \
  __device__ static __inline__ void operator+=(TYPE##2 &a, TYPE b) \
  { \
    a = make_##TYPE##2(a.x + b, a.y + b); \
  } \
  __device__ static __inline__ TYPE##3 operator+(TYPE##3 a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a.x + b.x, a.y + b.y, a.z + b.z); \
  } \
  __device__ static __inline__ void operator+=(TYPE##3 &a, TYPE##3 b) \
  { \
    a = make_##TYPE##3(a.x + b.x, a.y + b.y, a.z + b.z); \
  } \
  __device__ static __inline__ TYPE##3 operator+(TYPE##3 a, TYPE b) \
  { \
    return make_##TYPE##3(a.x + b, a.y + b, a.z + b); \
  } \
  __device__ static __inline__ TYPE##3 operator+(TYPE a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a + b.x, a + b.y, a + b.z); \
  } \
  __device__ static __inline__ void operator+=(TYPE##3 &a, TYPE b) \
  { \
    a = make_##TYPE##3(a.x + b, a.y + b, a.z + b); \
  } \
  __device__ static __inline__ TYPE##4 operator+(TYPE##4 a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); \
  } \
  __device__ static __inline__ void operator+=(TYPE##4 &a, TYPE##4 b) \
  { \
    a = make_##TYPE##4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); \
  } \
  __device__ static __inline__ TYPE##4 operator+(TYPE##4 a, TYPE b) \
  { \
    return make_##TYPE##4(a.x + b, a.y + b, a.z + b, a.w + b); \
  } \
  __device__ static __inline__ TYPE##4 operator+(TYPE a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a + b.x, a + b.y, a + b.z, a + b.w); \
  } \
  __device__ static __inline__ void operator+=(TYPE##4 &a, TYPE b) \
  { \
    a = make_##TYPE##4(a.x + b, a.y + b, a.z + b, a.w + b); \
  }

__MAKE_PLUS(float)
__MAKE_PLUS(int)
__MAKE_PLUS(uint)

#undef __MAKE_PLUS

#define __MAKE_MINUS(TYPE) \
  __device__ static __inline__ TYPE##2 operator-(TYPE##2 a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a.x - b.x, a.y - b.y); \
  } \
  __device__ static __inline__ void operator-=(TYPE##2 &a, TYPE##2 b) \
  { \
    a = make_##TYPE##2(a.x - b.x, a.y - b.y); \
  } \
  __device__ static __inline__ TYPE##2 operator-(TYPE##2 a, TYPE b) \
  { \
    return make_##TYPE##2(a.x - b, a.y - b); \
  } \
  __device__ static __inline__ TYPE##2 operator-(TYPE a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a - b.x, a - b.y); \
  } \
  __device__ static __inline__ void operator-=(TYPE##2 &a, TYPE b) \
  { \
    a = make_##TYPE##2(a.x - b, a.y - b); \
  } \
  __device__ static __inline__ TYPE##3 operator-(TYPE##3 a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a.x - b.x, a.y - b.y, a.z - b.z); \
  } \
  __device__ static __inline__ void operator-=(TYPE##3 &a, TYPE##3 b) \
  { \
    a = make_##TYPE##3(a.x - b.x, a.y - b.y, a.z - b.z); \
  } \
  __device__ static __inline__ TYPE##3 operator-(TYPE##3 a, TYPE b) \
  { \
    return make_##TYPE##3(a.x - b, a.y - b, a.z - b); \
  } \
  __device__ static __inline__ TYPE##3 operator-(TYPE a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a - b.x, a - b.y, a - b.z); \
  } \
  __device__ static __inline__ void operator-=(TYPE##3 &a, TYPE b) \
  { \
    a = make_##TYPE##3(a.x - b, a.y - b, a.z - b); \
  } \
  __device__ static __inline__ TYPE##4 operator-(TYPE##4 a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); \
  } \
  __device__ static __inline__ void operator-=(TYPE##4 &a, TYPE##4 b) \
  { \
    a = make_##TYPE##4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); \
  } \
  __device__ static __inline__ TYPE##4 operator-(TYPE##4 a, TYPE b) \
  { \
    return make_##TYPE##4(a.x - b, a.y - b, a.z - b, a.w - b); \
  } \
  __device__ static __inline__ TYPE##4 operator-(TYPE a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a - b.x, a - b.y, a - b.z, a - b.w); \
  } \
  __device__ static __inline__ void operator-=(TYPE##4 &a, TYPE b) \
  { \
    a = make_##TYPE##4(a.x - b, a.y - b, a.z - b, a.w - b); \
  }

__MAKE_MINUS(float)
__MAKE_MINUS(int)
__MAKE_MINUS(uint)

#undef __MAKE_MINUS

#define __MAKE_TIMES(TYPE) \
  __device__ static __inline__ TYPE##2 operator*(TYPE##2 a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a.x * b.x, a.y * b.y); \
  } \
  __device__ static __inline__ void operator*=(TYPE##2 &a, TYPE##2 b) \
  { \
    a = make_##TYPE##2(a.x * b.x, a.y * b.y); \
  } \
  __device__ static __inline__ TYPE##2 operator*(TYPE##2 a, TYPE b) \
  { \
    return make_##TYPE##2(a.x * b, a.y * b); \
  } \
  __device__ static __inline__ TYPE##2 operator*(TYPE a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a * b.x, a * b.y); \
  } \
  __device__ static __inline__ void operator*=(TYPE##2 &a, TYPE b) \
  { \
    a = make_##TYPE##2(a.x * b, a.y * b); \
  } \
  __device__ static __inline__ TYPE##3 operator*(TYPE##3 a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a.x * b.x, a.y * b.y, a.z * b.z); \
  } \
  __device__ static __inline__ void operator*=(TYPE##3 &a, TYPE##3 b) \
  { \
    a = make_##TYPE##3(a.x * b.x, a.y * b.y, a.z * b.z); \
  } \
  __device__ static __inline__ TYPE##3 operator*(TYPE##3 a, TYPE b) \
  { \
    return make_##TYPE##3(a.x * b, a.y * b, a.z * b); \
  } \
  __device__ static __inline__ TYPE##3 operator*(TYPE a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a * b.x, a * b.y, a * b.z); \
  } \
  __device__ static __inline__ void operator*=(TYPE##3 &a, TYPE b) \
  { \
    a = make_##TYPE##3(a.x * b, a.y * b, a.z * b); \
  } \
  __device__ static __inline__ TYPE##4 operator*(TYPE##4 a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); \
  } \
  __device__ static __inline__ void operator*=(TYPE##4 &a, TYPE##4 b) \
  { \
    a = make_##TYPE##4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); \
  } \
  __device__ static __inline__ TYPE##4 operator*(TYPE##4 a, TYPE b) \
  { \
    return make_##TYPE##4(a.x * b, a.y * b, a.z * b, a.w * b); \
  } \
  __device__ static __inline__ TYPE##4 operator*(TYPE a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a * b.x, a * b.y, a * b.z, a * b.w); \
  } \
  __device__ static __inline__ void operator*=(TYPE##4 &a, TYPE b) \
  { \
    a = make_##TYPE##4(a.x * b, a.y * b, a.z * b, a.w * b); \
  }

__MAKE_TIMES(float)
__MAKE_TIMES(int)
__MAKE_TIMES(uint)

#undef __MAKE_TIMES

#define __MAKE_DIVIDE(TYPE) \
  __device__ static __inline__ TYPE##2 operator/(TYPE##2 a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a.x / b.x, a.y / b.y); \
  } \
  __device__ static __inline__ void operator/=(TYPE##2 &a, TYPE##2 b) \
  { \
    a = make_##TYPE##2(a.x / b.x, a.y / b.y); \
  } \
  __device__ static __inline__ TYPE##2 operator/(TYPE##2 a, TYPE b) \
  { \
    return make_##TYPE##2(a.x / b, a.y / b); \
  } \
  __device__ static __inline__ TYPE##2 operator/(TYPE a, TYPE##2 b) \
  { \
    return make_##TYPE##2(a / b.x, a / b.y); \
  } \
  __device__ static __inline__ void operator/=(TYPE##2 &a, TYPE b) \
  { \
    a = make_##TYPE##2(a.x / b, a.y / b); \
  } \
  __device__ static __inline__ TYPE##3 operator/(TYPE##3 a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a.x / b.x, a.y / b.y, a.z / b.z); \
  } \
  __device__ static __inline__ void operator/=(TYPE##3 &a, TYPE##3 b) \
  { \
    a = make_##TYPE##3(a.x / b.x, a.y / b.y, a.z / b.z); \
  } \
  __device__ static __inline__ TYPE##3 operator/(TYPE##3 a, TYPE b) \
  { \
    return make_##TYPE##3(a.x / b, a.y / b, a.z / b); \
  } \
  __device__ static __inline__ TYPE##3 operator/(TYPE a, TYPE##3 b) \
  { \
    return make_##TYPE##3(a / b.x, a / b.y, a / b.z); \
  } \
  __device__ static __inline__ void operator/=(TYPE##3 &a, TYPE b) \
  { \
    a = make_##TYPE##3(a.x / b, a.y / b, a.z / b); \
  } \
  __device__ static __inline__ TYPE##4 operator/(TYPE##4 a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); \
  } \
  __device__ static __inline__ void operator/=(TYPE##4 &a, TYPE##4 b) \
  { \
    a = make_##TYPE##4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); \
  } \
  __device__ static __inline__ TYPE##4 operator/(TYPE##4 a, TYPE b) \
  { \
    return make_##TYPE##4(a.x / b, a.y / b, a.z / b, a.w / b); \
  } \
  __device__ static __inline__ TYPE##4 operator/(TYPE a, TYPE##4 b) \
  { \
    return make_##TYPE##4(a / b.x, a / b.y, a / b.z, a / b.w); \
  } \
  __device__ static __inline__ void operator/=(TYPE##4 &a, TYPE b) \
  { \
    a = make_##TYPE##4(a.x / b, a.y / b, a.z / b, a.w / b); \
  }

__MAKE_DIVIDE(float)

#undef __MAKE_DIVIDE

 __device__ float2 fminf(float2 a, float2 b);
__device__ float3 fminf(float3 a, float3 b);
 __device__ float4 fminf(float4 a, float4 b);

__device__ int2 min(int2 a, int2 b);
__device__ int3 min(int3 a, int3 b);
__device__ int4 min(int4 a, int4 b);

__device__ uint2 min(uint2 a, uint2 b);
__device__ uint3 min(uint3 a, uint3 b);
__device__ uint4 min(uint4 a, uint4 b);

__device__ float2 fmaxf(float2 a, float2 b);
__device__ float3 fmaxf(float3 a, float3 b);
__device__ float4 fmaxf(float4 a, float4 b);

__device__ int2 max(int2 a, int2 b);
__device__ int3 max(int3 a, int3 b);
__device__ int4 max(int4 a, int4 b);

__device__ uint2 max(uint2 a, uint2 b);
__device__ uint3 max(uint3 a, uint3 b);
__device__ uint4 max(uint4 a, uint4 b);

__device__ float lerp(float a, float b, float t);
__device__ float2 lerp(float2 a, float2 b, float t);
__device__ float3 lerp(float3 a, float3 b, float t);
__device__ float4 lerp(float4 a, float4 b, float t);

__device__ float clamp(float f, float a, float b);
__device__ int clamp(int f, int a, int b);
__device__ uint clamp(uint f, uint a, uint b);

__device__ float2 clamp(float2 v, float a, float b);
__device__ float2 clamp(float2 v, float2 a, float2 b);
__device__ float3 clamp(float3 v, float a, float b);
__device__ float3 clamp(float3 v, float3 a, float3 b);
__device__ float4 clamp(float4 v, float a, float b);
__device__ float4 clamp(float4 v, float4 a, float4 b);

__device__ int2 clamp(int2 v, int a, int b);
__device__ int2 clamp(int2 v, int2 a, int2 b);
__device__ int3 clamp(int3 v, int a, int b);
__device__ int3 clamp(int3 v, int3 a, int3 b);
__device__ int4 clamp(int4 v, int a, int b);
__device__ int4 clamp(int4 v, int4 a, int4 b);

__device__ uint2 clamp(uint2 v, uint a, uint b);
__device__ uint2 clamp(uint2 v, uint2 a, uint2 b);
__device__ uint3 clamp(uint3 v, uint a, uint b);
__device__ uint3 clamp(uint3 v, uint3 a, uint3 b);
__device__ uint4 clamp(uint4 v, uint a, uint b);
__device__ uint4 clamp(uint4 v, uint4 a, uint4 b);

__device__ float dot(float2 a, float2 b);
__device__ float dot(float3 a, float3 b);
__device__ float dot(float4 a, float4 b);

__device__ int dot(int2 a, int2 b);
__device__ int dot(int3 a, int3 b);
__device__ int dot(int4 a, int4 b);

__device__ uint dot(uint2 a, uint2 b);
__device__ uint dot(uint3 a, uint3 b);
__device__ uint dot(uint4 a, uint4 b);

__device__ float length(float2 v);
__device__ float length(float3 v);
__device__ float length(float4 v);

__device__ float2 normalize(float2 v);
__device__ float3 normalize(float3 v);
__device__ float4 normalize(float4 v);

__device__ float2 floorf(float2 v);
__device__ float3 floorf(float3 v);
__device__ float4 floorf(float4 v);

__device__ float fracf(float v);
__device__ float2 fracf(float2 v);
__device__ float3 fracf(float3 v);
__device__ float4 fracf(float4 v);

__device__ float2 fmodf(float2 a, float2 b);
__device__ float3 fmodf(float3 a, float3 b);
__device__ float4 fmodf(float4 a, float4 b);

__device__ float2 fabs(float2 v);
__device__ float3 fabs(float3 v);
__device__ float4 fabs(float4 v);

__device__ int2 abs(int2 v);
__device__ int3 abs(int3 v);
__device__ int4 abs(int4 v);

__device__ float3 reflect(float3 i, float3 n);
__device__ float3 cross(float3 a, float3 b);

__device__ float smoothstep(float a, float b, float x);
__device__ float2 smoothstep(float2 a, float2 b, float2 x);
__device__ float3 smoothstep(float3 a, float3 b, float3 x);
__device__ float4 smoothstep(float4 a, float4 b, float4 x);

#pragma GCC diagnostic pop

#endif
