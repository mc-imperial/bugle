/* See Table B-1 in CUDA Specification */
typedef char char1 __attribute__((ext_vector_type(1)));
typedef char char2 __attribute__((ext_vector_type(2)));
typedef char char3 __attribute__((ext_vector_type(3)));
typedef char char4 __attribute__((ext_vector_type(4)));
typedef unsigned char uchar1 __attribute__((ext_vector_type(1)));
typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
typedef unsigned char uchar3 __attribute__((ext_vector_type(3)));
typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));

typedef short short1 __attribute__((ext_vector_type(1)));
typedef short short2 __attribute__((ext_vector_type(2)));
typedef short short3 __attribute__((ext_vector_type(3)));
typedef short short4 __attribute__((ext_vector_type(4)));
typedef unsigned short ushort1 __attribute__((ext_vector_type(1)));
typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
typedef unsigned short ushort3 __attribute__((ext_vector_type(3)));
typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));

typedef int int1 __attribute__((ext_vector_type(1)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint1 __attribute__((ext_vector_type(1)));
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef unsigned int uint3 __attribute__((ext_vector_type(3)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

typedef long long1 __attribute__((ext_vector_type(1)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef long long3 __attribute__((ext_vector_type(3)));
typedef long long4 __attribute__((ext_vector_type(4)));
typedef unsigned long ulong1 __attribute__((ext_vector_type(1)));
typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
typedef unsigned long ulong3 __attribute__((ext_vector_type(3)));
typedef unsigned long ulong4 __attribute__((ext_vector_type(4)));

typedef long long int longlong1 __attribute__((ext_vector_type(1)));
typedef long long int longlong2 __attribute__((ext_vector_type(2)));
typedef long long int longlong3 __attribute__((ext_vector_type(3)));
typedef long long int longlong4 __attribute__((ext_vector_type(4)));
typedef unsigned long long int ulonglong1 __attribute__((ext_vector_type(1)));
typedef unsigned long long int ulonglong2 __attribute__((ext_vector_type(2)));
typedef unsigned long long int ulonglong3 __attribute__((ext_vector_type(3)));
typedef unsigned long long int ulonglong4 __attribute__((ext_vector_type(4)));

typedef float float1 __attribute__((ext_vector_type(1)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

typedef double double1 __attribute__((ext_vector_type(1)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));
typedef double double4 __attribute__((ext_vector_type(4)));

/* From vector_functions.h */

__device__ static __inline__ char1 make_char1(signed char x)
{
  char1 t; t.x = x; return t;
}

__device__ static __inline__ uchar1 make_uchar1(unsigned char x)
{
  uchar1 t; t.x = x; return t;
}

__device__ static __inline__ char2 make_char2(signed char x, signed char y)
{
  char2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ char2 make_char2(signed char x)
{
  return make_char2(x,x);
}

__device__ static __inline__ uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ uchar2 make_uchar2(unsigned char x)
{
  return make_uchar2(x,x);
}

__device__ static __inline__ char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ char3 make_char3(signed char x)
{
  return make_char3(x,x,x);
}

__device__ static __inline__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ uchar3 make_uchar3(unsigned char x)
{
  return make_uchar3(x,x,x);
}

__device__ static __inline__ char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ char4 make_char4(signed char x)
{
  return make_char4(x,x,x,x);
}

__device__ static __inline__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ uchar4 make_uchar4(unsigned char x)
{
  return make_uchar4(x,x,x,x);
}

__device__ static __inline__ short1 make_short1(short x)
{
  short1 t; t.x = x; return t;
}

__device__ static __inline__ ushort1 make_ushort1(unsigned short x)
{
  ushort1 t; t.x = x; return t;
}

__device__ static __inline__ short2 make_short2(short x, short y)
{
  short2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ short2 make_short2(short x)
{
  return make_short2(x,x);
}

__device__ static __inline__ ushort2 make_ushort2(unsigned short x, unsigned short y)
{
  ushort2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ ushort2 make_ushort2(unsigned short x)
{
  return make_ushort2(x,x);
}

__device__ static __inline__ short3 make_short3(short x,short y, short z)
{
  short3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ short3 make_short3(short x)
{
  return make_short3(x,x,x);
}

__device__ static __inline__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
  ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ ushort3 make_ushort3(unsigned short x)
{
  return make_ushort3(x,x,x);
}

__device__ static __inline__ short4 make_short4(short x, short y, short z, short w)
{
  short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ short4 make_short4(short x)
{
  return make_short4(x,x,x,x);
}

__device__ static __inline__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
  ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ ushort4 make_ushort4(unsigned short x)
{
  return make_ushort4(x,x,x,x);
}

__device__ static __inline__ int1 make_int1(int x)
{
  int1 t; t.x = x; return t;
}

__device__ static __inline__ uint1 make_uint1(unsigned int x)
{
  uint1 t; t.x = x; return t;
}

__device__ static __inline__ int2 make_int2(int x, int y)
{
  int2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ int2 make_int2(int x)
{
  return make_int2(x,x);
}

__device__ static __inline__ uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ uint2 make_uint2(unsigned int x)
{
  return make_uint2(x,x);
}

__device__ static __inline__ int3 make_int3(int x, int y, int z)
{
  int3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ int3 make_int3(int x)
{
  return make_int3(x,x,x);
}

__device__ static __inline__ int3 make_int3(float3 a)
{
      return make_int3(int(a.x), int(a.y), int(a.z));
}

__device__ static __inline__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ uint3 make_uint3(unsigned int x)
{
  return make_uint3(x,x,x);
}

__device__ static __inline__ uint3 make_uint3(float3 a)
{
      return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}

__device__ static __inline__ int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ int4 make_int4(int x)
{
  return make_int4(x,x,x,x);
}

__device__ static __inline__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ uint4 make_uint4(unsigned int x)
{
  return make_uint4(x,x,x,x);
}

__device__ static __inline__ long1 make_long1(long int x)
{
  long1 t; t.x = x; return t;
}

__device__ static __inline__ ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t; t.x = x; return t;
}

__device__ static __inline__ long2 make_long2(long int x, long int y)
{
  long2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ long2 make_long2(long int x)
{
  return make_long2(x,x);
}

__device__ static __inline__ ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ ulong2 make_ulong2(unsigned long int x)
{
  return make_ulong2(x,x);
}

__device__ static __inline__ long3 make_long3(long int x, long int y, long int z)
{
  long3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ long3 make_long3(long int x)
{
  return make_long3(x,x,x);
}

__device__ static __inline__ ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ ulong3 make_ulong3(unsigned long int x)
{
  return make_ulong3(x,x,x);
}

__device__ static __inline__ long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ long4 make_long4(long int x)
{
  return make_long4(x,x,x,x);
}

__device__ static __inline__ ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ ulong4 make_ulong4(unsigned long int x)
{
  return make_ulong4(x,x,x,x);
}

__device__ static __inline__ float1 make_float1(float x)
{
  float1 t; t.x = x; return t;
}

__device__ static __inline__ float2 make_float2(float x, float y)
{
  float2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ float2 make_float2(float x)
{
  return make_float2(x,x);
}

__device__ static __inline__ float2 make_float2(int2 a)
{
  return make_float2(float(a.x),float(a.y));
}

__device__ static __inline__ float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ float3 make_float3(float x)
{
  return make_float3(x,x,x);
}

__device__ static __inline__ float3 make_float3(float2 a)
{
      return make_float3(a.x, a.y, 0.0f);
}
__device__ static __inline__ float3 make_float3(float2 a, float s)
{
      return make_float3(a.x, a.y, s);
}
__device__ static __inline__ float3 make_float3(float4 a)
{
      return make_float3(a.x, a.y, a.z);  // discards w
}
__device__ static __inline__ float3 make_float3(int3 a)
{
      return make_float3(float(a.x), float(a.y), float(a.z));
}

__device__ static __inline__ float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ float4 make_float4(float x)
{
  return make_float4(x,x,x,x);
}

__device__ static __inline__ float4 make_float4(float3 a)
{
      return make_float4(a.x, a.y, a.z, 0.0f);
}
__device__ static __inline__ float4 make_float4(float3 a, float w)
{
      return make_float4(a.x, a.y, a.z, w);
}
__device__ static __inline__ float4 make_float4(int4 a)
{
      return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

__device__ static __inline__ longlong1 make_longlong1(long long int x)
{
  longlong1 t; t.x = x; return t;
}

__device__ static __inline__ ulonglong1 make_ulonglong1(unsigned long long int x)
{
  ulonglong1 t; t.x = x; return t;
}

__device__ static __inline__ longlong2 make_longlong2(long long int x, long long int y)
{
  longlong2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ longlong2 make_longlong2(long long int x)
{
  return make_longlong2(x,x);
}

__device__ static __inline__ ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
  ulonglong2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ ulonglong2 make_ulonglong2(unsigned long long int x)
{
  return make_ulonglong2(x,x);
}

__device__ static __inline__ longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
  longlong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ longlong3 make_longlong3(long long int x)
{
  return make_longlong3(x,x,x);
}

__device__ static __inline__ ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
  ulonglong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ ulonglong3 make_ulonglong3(unsigned long long int x)
{
  return make_ulonglong3(x,x,x);
}

__device__ static __inline__ longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
  longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ longlong4 make_longlong4(long long int x)
{
  return make_longlong4(x,x,x,x);
}

__device__ static __inline__ ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
  ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ ulonglong4 make_ulonglong4(unsigned long long int x)
{
  return make_ulonglong4(x,x,x,x);
}

__device__ static __inline__ double1 make_double1(double x)
{
  double1 t; t.x = x; return t;
}

__device__ static __inline__ double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

__device__ static __inline__ double2 make_double2(double x)
{
  return make_double2(x,x);
}

__device__ static __inline__ double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ static __inline__ double3 make_double3(double x)
{
  return make_double3(x,x,x);
}

__device__ static __inline__ double4 make_double4(double x, double y, double z, double w)
{
  double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__device__ static __inline__ double4 make_double4(double x)
{
  return make_double4(x,x,x,x);
}
