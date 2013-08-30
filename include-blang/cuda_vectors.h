/* See Table B-1 in CUDA Specification */
typedef struct {
  char x;
} char1;
typedef struct {
  char x, y;
} char2;
typedef struct {
  char x, y, z;
} char3;
typedef struct {
  char x, y, z, w;
} char4;
typedef struct {
  unsigned char x;
} uchar1;
typedef struct {
  unsigned char x, y;
} uchar2;
typedef struct {
  unsigned char x, y, z;
} uchar3;
typedef struct {
  unsigned char x, y, z, w;
} uchar4;

typedef struct {
  short x;
} short1;
typedef struct {
  short x, y;
} short2;
typedef struct {
  short x, y, z;
} short3;
typedef struct {
  short x, y, z, w;
} short4;
typedef struct {
  unsigned short x;
} ushort1;
typedef struct {
  unsigned short x, y;
} ushort2;
typedef struct {
  unsigned short x, y, z;
} ushort3;
typedef struct {
  unsigned short x, y, z, w;
} ushort4;

typedef struct {
  int x;
} int1;
typedef struct {
  int x, y;
} int2;
typedef struct {
  int x, y, z;
} int3;
typedef struct {
  int x, y, z, w;
} int4;
typedef struct {
  unsigned int x;
} uint1;
typedef struct {
  unsigned int x, y;
} uint2;
typedef struct {
  unsigned int x, y, z;
} uint3;
typedef struct {
  unsigned int x, y, z, w;
} uint4;

typedef struct {
  long x;
} long1;
typedef struct {
  long x, y;
} long2;
typedef struct {
  long x, y, z;
} long3;
typedef struct {
  long x, y, z, w;
} long4;
typedef struct {
  unsigned long x;
} ulong1;
typedef struct {
  unsigned long x, y;
} ulong2;
typedef struct {
  unsigned long x, y, z;
} ulong3;
typedef struct {
  unsigned long x, y, z, w;
} ulong4;

typedef struct {
  long long x;
} longlong1;
typedef struct {
  long long x, y;
} longlong2;
typedef struct {
  long long x, y, z;
} longlong3;
typedef struct {
  long long x, y, z, w;
} longlong4;
typedef struct {
  unsigned long long x;
} ulonglong1;
typedef struct {
  unsigned long long x, y;
} ulonglong2;
typedef struct {
  unsigned long long x, y, z;
} ulonglong3;
typedef struct {
  unsigned long long x, y, z, w;
} ulonglong4;

typedef struct {
  float x;
} float1;
typedef struct {
  float x, y;
} float2;
typedef struct {
  float x, y, z;
} float3;
typedef struct {
  float x, y, z, w;
} float4;

typedef struct {
  double x;
} double1;
typedef struct {
  double x, y;
} double2;
typedef struct {
  double x, y, z;
} double3;
typedef struct {
  double x, y, z, w;
} double4;

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

typedef unsigned int uint;

/* from helper_math.h */

__device__ float2 make_float2(float s);
__device__ float2 make_float2(float3 a);
__device__ float2 make_float2(int2 a);
__device__ float2 make_float2(uint2 a);

__device__ int2 make_int2(int s);
__device__ int2 make_int2(int3 a);
__device__ int2 make_int2(uint2 a);
__device__ int2 make_int2(float2 a);

__device__ uint2 make_uint2(uint s);
__device__ uint2 make_uint2(uint3 a);
__device__ uint2 make_uint2(int2 a);

__device__ float3 make_float3(float s);
__device__ float3 make_float3(float2 a);
__device__ float3 make_float3(float2 a, float s);
__device__ float3 make_float3(float4 a);
__device__ float3 make_float3(int3 a);
__device__ float3 make_float3(uint3 a);

__device__ int3 make_int3(int s);
__device__ int3 make_int3(int2 a);
__device__ int3 make_int3(int2 a, int s);
__device__ int3 make_int3(uint3 a);
__device__ int3 make_int3(float3 a);

__device__ uint3 make_uint3(uint s);
__device__ uint3 make_uint3(uint2 a);
__device__ uint3 make_uint3(uint2 a, uint s);
__device__ uint3 make_uint3(uint4 a);
__device__ uint3 make_uint3(int3 a);

__device__ float4 make_float4(float s);
__device__ float4 make_float4(float3 a);
__device__ float4 make_float4(float3 a, float w);
__device__ float4 make_float4(int4 a);
__device__ float4 make_float4(uint4 a);

__device__ int4 make_int4(int s);
__device__ int4 make_int4(int3 a);
__device__ int4 make_int4(int3 a, int w);
__device__ int4 make_int4(uint4 a);
__device__ int4 make_int4(float4 a);

__device__ uint4 make_uint4(uint s);
__device__ uint4 make_uint4(uint3 a);
__device__ uint4 make_uint4(uint3 a, uint w);
__device__ uint4 make_uint4(int4 a);

__device__ float2 operator-(float2 &a);
__device__ int2 operator-(int2 &a);
__device__ float3 operator-(float3 &a);
__device__ int3 operator-(int3 &a);
__device__ float4 operator-(float4 &a);
__device__ int4 operator-(int4 &a);

__device__ float2 operator+(float2 a, float2 b);
__device__ void operator+=(float2 &a, float2 b);
__device__ float2 operator+(float2 a, float b);
__device__ float2 operator+(float b, float2 a);
__device__ void operator+=(float2 &a, float b);

__device__ int2 operator+(int2 a, int2 b);
__device__ void operator+=(int2 &a, int2 b);
__device__ int2 operator+(int2 a, int b);
__device__ int2 operator+(int b, int2 a);
__device__ void operator+=(int2 &a, int b);

__device__ uint2 operator+(uint2 a, uint2 b);
__device__ void operator+=(uint2 &a, uint2 b);
__device__ uint2 operator+(uint2 a, uint b);
__device__ uint2 operator+(uint b, uint2 a);
__device__ void operator+=(uint2 &a, uint b);


__device__ float3 operator+(float3 a, float3 b);
__device__ void operator+=(float3 &a, float3 b);
__device__ float3 operator+(float3 a, float b);
__device__ void operator+=(float3 &a, float b);

__device__ int3 operator+(int3 a, int3 b);
__device__ void operator+=(int3 &a, int3 b);
__device__ int3 operator+(int3 a, int b);
__device__ void operator+=(int3 &a, int b);

__device__ uint3 operator+(uint3 a, uint3 b);
__device__ void operator+=(uint3 &a, uint3 b);
__device__ uint3 operator+(uint3 a, uint b);
__device__ void operator+=(uint3 &a, uint b);

__device__ int3 operator+(int b, int3 a);
__device__ uint3 operator+(uint b, uint3 a);
__device__ float3 operator+(float b, float3 a);

__device__ float4 operator+(float4 a, float4 b);
__device__ void operator+=(float4 &a, float4 b);
__device__ float4 operator+(float4 a, float b);
__device__ float4 operator+(float b, float4 a);
__device__ void operator+=(float4 &a, float b);

__device__ int4 operator+(int4 a, int4 b);
__device__ void operator+=(int4 &a, int4 b);
__device__ int4 operator+(int4 a, int b);
__device__ int4 operator+(int b, int4 a);
__device__ void operator+=(int4 &a, int b);

__device__ uint4 operator+(uint4 a, uint4 b);
__device__ void operator+=(uint4 &a, uint4 b);
__device__ uint4 operator+(uint4 a, uint b);
__device__ uint4 operator+(uint b, uint4 a);
__device__ void operator+=(uint4 &a, uint b);

__device__ float2 operator-(float2 a, float2 b);
__device__ void operator-=(float2 &a, float2 b);
__device__ float2 operator-(float2 a, float b);
__device__ float2 operator-(float b, float2 a);
__device__ void operator-=(float2 &a, float b);

__device__ int2 operator-(int2 a, int2 b);
__device__ void operator-=(int2 &a, int2 b);
__device__ int2 operator-(int2 a, int b);
__device__ int2 operator-(int b, int2 a);
__device__ void operator-=(int2 &a, int b);

__device__ uint2 operator-(uint2 a, uint2 b);
__device__ void operator-=(uint2 &a, uint2 b);
__device__ uint2 operator-(uint2 a, uint b);
__device__ uint2 operator-(uint b, uint2 a);
__device__ void operator-=(uint2 &a, uint b);

__device__ float3 operator-(float3 a, float3 b);
__device__ void operator-=(float3 &a, float3 b);
__device__ float3 operator-(float3 a, float b);
__device__ float3 operator-(float b, float3 a);
__device__ void operator-=(float3 &a, float b);

__device__ int3 operator-(int3 a, int3 b);
__device__ void operator-=(int3 &a, int3 b);
__device__ int3 operator-(int3 a, int b);
__device__ int3 operator-(int b, int3 a);
__device__ void operator-=(int3 &a, int b);

__device__ uint3 operator-(uint3 a, uint3 b);
__device__ void operator-=(uint3 &a, uint3 b);
__device__ uint3 operator-(uint3 a, uint b);
__device__ uint3 operator-(uint b, uint3 a);
__device__ void operator-=(uint3 &a, uint b);

__device__ float4 operator-(float4 a, float4 b);
__device__ void operator-=(float4 &a, float4 b);
__device__ float4 operator-(float4 a, float b);
__device__ void operator-=(float4 &a, float b);

__device__ int4 operator-(int4 a, int4 b);
__device__ void operator-=(int4 &a, int4 b);
__device__ int4 operator-(int4 a, int b);
__device__ int4 operator-(int b, int4 a);
__device__ void operator-=(int4 &a, int b);

__device__ uint4 operator-(uint4 a, uint4 b);
__device__ void operator-=(uint4 &a, uint4 b);
__device__ uint4 operator-(uint4 a, uint b);
__device__ uint4 operator-(uint b, uint4 a);
__device__ void operator-=(uint4 &a, uint b);

__device__ float2 operator*(float2 a, float2 b);
__device__ void operator*=(float2 &a, float2 b);
__device__ float2 operator*(float2 a, float b);
__device__ float2 operator*(float b, float2 a);
__device__ void operator*=(float2 &a, float b);

__device__ int2 operator*(int2 a, int2 b);
__device__ void operator*=(int2 &a, int2 b);
__device__ int2 operator*(int2 a, int b);
__device__ int2 operator*(int b, int2 a);
__device__ void operator*=(int2 &a, int b);

__device__ uint2 operator*(uint2 a, uint2 b);
__device__ void operator*=(uint2 &a, uint2 b);
__device__ uint2 operator*(uint2 a, uint b);
__device__ uint2 operator*(uint b, uint2 a);
__device__ void operator*=(uint2 &a, uint b);

__device__ float3 operator*(float3 a, float3 b);
__device__ void operator*=(float3 &a, float3 b);
__device__ float3 operator*(float3 a, float b);
__device__ float3 operator*(float b, float3 a);
__device__ void operator*=(float3 &a, float b);

__device__ int3 operator*(int3 a, int3 b);
__device__ void operator*=(int3 &a, int3 b);
__device__ int3 operator*(int3 a, int b);
__device__ int3 operator*(int b, int3 a);
__device__ void operator*=(int3 &a, int b);

__device__ uint3 operator*(uint3 a, uint3 b);
__device__ void operator*=(uint3 &a, uint3 b);
__device__ uint3 operator*(uint3 a, uint b);
__device__ uint3 operator*(uint b, uint3 a);
__device__ void operator*=(uint3 &a, uint b);

__device__ float4 operator*(float4 a, float4 b);
__device__ void operator*=(float4 &a, float4 b);
__device__ float4 operator*(float4 a, float b);
__device__ float4 operator*(float b, float4 a);
__device__ void operator*=(float4 &a, float b);

__device__ int4 operator*(int4 a, int4 b);
__device__ void operator*=(int4 &a, int4 b);
__device__ int4 operator*(int4 a, int b);
__device__ int4 operator*(int b, int4 a);
__device__ void operator*=(int4 &a, int b);

__device__ uint4 operator*(uint4 a, uint4 b);
__device__ void operator*=(uint4 &a, uint4 b);
__device__ uint4 operator*(uint4 a, uint b);
__device__ uint4 operator*(uint b, uint4 a);
__device__ void operator*=(uint4 &a, uint b);

__device__ float2 operator/(float2 a, float2 b);
__device__ void operator/=(float2 &a, float2 b);
__device__ float2 operator/(float2 a, float b);
__device__ void operator/=(float2 &a, float b);
__device__ float2 operator/(float b, float2 a);

__device__ float3 operator/(float3 a, float3 b);
__device__ void operator/=(float3 &a, float3 b);
__device__ float3 operator/(float3 a, float b);
__device__ void operator/=(float3 &a, float b);
__device__ float3 operator/(float b, float3 a);

__device__ float4 operator/(float4 a, float4 b);
__device__ void operator/=(float4 &a, float4 b);
__device__ float4 operator/(float4 a, float b);
__device__ void operator/=(float4 &a, float b);
__device__ float4 operator/(float b, float4 a);

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
