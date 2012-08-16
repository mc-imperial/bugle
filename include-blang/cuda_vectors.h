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
typedef unsigned long long int ulonglong1 __attribute__((ext_vector_type(1)));
typedef unsigned long long int ulonglong2 __attribute__((ext_vector_type(2)));

typedef float float1 __attribute__((ext_vector_type(1)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

typedef double double1 __attribute__((ext_vector_type(1)));
typedef double double2 __attribute__((ext_vector_type(2)));
