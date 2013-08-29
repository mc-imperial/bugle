#ifndef CUDA_TEXTURES_H
#define CUDA_TEXTURES_H

#define cudaTextureType1D              0x01
#define cudaTextureType2D              0x02
#define cudaTextureType3D              0x03
#define cudaTextureTypeCubemap         0x0C
#define cudaTextureType1DLayered       0xF1
#define cudaTextureType2DLayered       0xF2
#define cudaTextureTypeCubemapLayered  0xFC
enum cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,
    cudaAddressModeClamp  = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
};
enum cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,
    cudaFilterModeLinear = 1
};
enum cudaTextureReadMode
{
    cudaReadModeElementType     = 0,
    cudaReadModeNormalizedFloat = 1
};

typedef unsigned long long cudaTextureObject_t;

template<class T, int texType = cudaTextureType1D, enum cudaTextureReadMode mode = cudaReadModeElementType>
struct texture { T a; int b; enum cudaTextureReadMode c; };

/* texture_fetch_instructions.h */

template <class T>
__device__ T tex1Dfetch(texture<T, cudaTextureType1D, cudaReadModeElementType> t, int x);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1Dfetch(texture<TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float tex1Dfetch(texture<unsigned TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float1 tex1Dfetch(texture<TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float1 tex1Dfetch(texture<u##TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float2 tex1Dfetch(texture<TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float2 tex1Dfetch(texture<u##TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float4 tex1Dfetch(texture<TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x); \
  __device__ float4 tex1Dfetch(texture<u##TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex1D(texture<T, cudaTextureType1D, cudaReadModeElementType> t, float x);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1D(texture<TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float tex1D(texture<unsigned TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float1 tex1D(texture<TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float1 tex1D(texture<u##TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float2 tex1D(texture<TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float2 tex1D(texture<u##TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float4 tex1D(texture<TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x); \
  __device__ float4 tex1D(texture<u##TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2D(texture<T, cudaTextureType2D, cudaReadModeElementType> t, float x, float y);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2D(texture<TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float tex2D(texture<unsigned TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float1 tex2D(texture<TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float1 tex2D(texture<u##TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float2 tex2D(texture<TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float2 tex2D(texture<u##TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float4 tex2D(texture<TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y); \
  __device__ float4 tex2D(texture<u##TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex1DLayered(texture<T, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1DLayered(texture<TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float tex1DLayered(texture<unsigned TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float1 tex1DLayered(texture<TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float1 tex1DLayered(texture<u##TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float2 tex1DLayered(texture<TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float2 tex1DLayered(texture<u##TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float4 tex1DLayered(texture<TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer); \
  __device__ float4 tex1DLayered(texture<u##TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2DLayered(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2DLayered(texture<TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float tex2DLayered(texture<unsigned TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float1 tex2DLayered(texture<TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float1 tex2DLayered(texture<u##TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float2 tex2DLayered(texture<TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float2 tex2DLayered(texture<u##TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float4 tex2DLayered(texture<TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer); \
  __device__ float4 tex2DLayered(texture<u##TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex3D(texture<T, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex3D(texture<TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float tex3D(texture<unsigned TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float1 tex3D(texture<TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float1 tex3D(texture<u##TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float2 tex3D(texture<TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float2 tex3D(texture<u##TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float4 tex3D(texture<TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float4 tex3D(texture<u##TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T texCubemap(texture<T, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z);
#define MK_GETFLOAT(TYPE) \
  __device__ float texCubemap(texture<TYPE, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float texCubemap(texture<unsigned TYPE, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float1 texCubemap(texture<TYPE##1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float1 texCubemap(texture<u##TYPE##1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float2 texCubemap(texture<TYPE##2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float2 texCubemap(texture<u##TYPE##2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float4 texCubemap(texture<TYPE##4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z); \
  __device__ float4 texCubemap(texture<u##TYPE##4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T texCubemapLayered(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer);
#define MK_GETFLOAT(TYPE) \
  __device__ float texCubemapLayered(texture<TYPE, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float texCubemapLayered(texture<unsigned TYPE, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float1 texCubemapLayered(texture<TYPE##1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float1 texCubemapLayered(texture<u##TYPE##1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float2 texCubemapLayered(texture<TYPE##2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float2 texCubemapLayered(texture<u##TYPE##2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float4 texCubemapLayered(texture<TYPE##4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer); \
  __device__ float4 texCubemapLayered(texture<u##TYPE##4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

__device__ char4 tex2Dgather(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ char4 tex2Dgather(texture<signed char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uchar4 tex2Dgather(texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ char4 tex2Dgather(texture<char1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uchar4 tex2Dgather(texture<uchar1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ char4 tex2Dgather(texture<char2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uchar4 tex2Dgather(texture<uchar2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ char4 tex2Dgather(texture<char3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uchar4 tex2Dgather(texture<uchar3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ char4 tex2Dgather(texture<char4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uchar4 tex2Dgather(texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ short4 tex2Dgather(texture<signed short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ ushort4 tex2Dgather(texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ short4 tex2Dgather(texture<short1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ ushort4 tex2Dgather(texture<ushort1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ short4 tex2Dgather(texture<short2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ ushort4 tex2Dgather(texture<ushort2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ short4 tex2Dgather(texture<short3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ ushort4 tex2Dgather(texture<ushort3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ short4 tex2Dgather(texture<short4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ ushort4 tex2Dgather(texture<ushort4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ int4 tex2Dgather(texture<signed int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uint4 tex2Dgather(texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ int4 tex2Dgather(texture<int1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uint4 tex2Dgather(texture<uint1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ int4 tex2Dgather(texture<int2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uint4 tex2Dgather(texture<uint2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ int4 tex2Dgather(texture<int3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uint4 tex2Dgather(texture<uint3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ int4 tex2Dgather(texture<int4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ uint4 tex2Dgather(texture<uint4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<float, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<float1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<float2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<float3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<float4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<signed char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<char1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<char2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<uchar2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<char3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<uchar3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<char4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<signed short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<short1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<ushort1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<short2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<ushort2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<short3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<ushort3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<short4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);
__device__ float4 tex2Dgather(texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0);

template <class T>
__device__ T tex1DLod(texture<T, cudaTextureType1D, cudaReadModeElementType> t, float x, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1DLod(texture<TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float tex1DLod(texture<unsigned TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float1 tex1DLod(texture<TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float1 tex1DLod(texture<u##TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float2 tex1DLod(texture<TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float2 tex1DLod(texture<u##TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float4 tex1DLod(texture<TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level); \
  __device__ float4 tex1DLod(texture<u##TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2DLod(texture<T, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2DLod(texture<TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float tex2DLod(texture<unsigned TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float1 tex2DLod(texture<TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float1 tex2DLod(texture<u##TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float2 tex2DLod(texture<TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float2 tex2DLod(texture<u##TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float4 tex2DLod(texture<TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level); \
  __device__ float4 tex2DLod(texture<u##TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex1DLayeredLod(texture<T, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1DLayeredLod(texture<TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float tex1DLayeredLod(texture<unsigned TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float1 tex1DLayeredLod(texture<TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float1 tex1DLayeredLod(texture<u##TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float2 tex1DLayeredLod(texture<TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float2 tex1DLayeredLod(texture<u##TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float4 tex1DLayeredLod(texture<TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level); \
  __device__ float4 tex1DLayeredLod(texture<u##TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2DLayeredLod(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2DLayeredLod(texture<TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float tex2DLayeredLod(texture<unsigned TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float1 tex2DLayeredLod(texture<TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float1 tex2DLayeredLod(texture<u##TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float2 tex2DLayeredLod(texture<TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float2 tex2DLayeredLod(texture<u##TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float4 tex2DLayeredLod(texture<TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level); \
  __device__ float4 tex2DLayeredLod(texture<u##TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex3DLod(texture<T, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex3DLod(texture<TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float tex3DLod(texture<unsigned TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float1 tex3DLod(texture<TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float1 tex3DLod(texture<u##TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float2 tex3DLod(texture<TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float2 tex3DLod(texture<u##TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float4 tex3DLod(texture<TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float4 tex3DLod(texture<u##TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T texCubemapLod(texture<T, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float texCubemapLod(texture<TYPE, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float texCubemapLod(texture<unsigned TYPE, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float1 texCubemapLod(texture<TYPE##1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float1 texCubemapLod(texture<u##TYPE##1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float2 texCubemapLod(texture<TYPE##2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float2 texCubemapLod(texture<u##TYPE##2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float4 texCubemapLod(texture<TYPE##4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level); \
  __device__ float4 texCubemapLod(texture<u##TYPE##4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T texCubemapLayeredLod(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level);
#define MK_GETFLOAT(TYPE) \
  __device__ float texCubemapLayeredLod(texture<TYPE, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float texCubemapLayeredLod(texture<unsigned TYPE, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float1 texCubemapLayeredLod(texture<TYPE##1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float1 texCubemapLayeredLod(texture<u##TYPE##1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float2 texCubemapLayeredLod(texture<TYPE##2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float2 texCubemapLayeredLod(texture<u##TYPE##2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float4 texCubemapLayeredLod(texture<TYPE##4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level); \
  __device__ float4 texCubemapLayeredLod(texture<u##TYPE##4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex1DGrad(texture<T, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1DGrad(texture<TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float tex1DGrad(texture<unsigned TYPE, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float1 tex1DGrad(texture<TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float1 tex1DGrad(texture<u##TYPE##1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float2 tex1DGrad(texture<TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float2 tex1DGrad(texture<u##TYPE##2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float4 tex1DGrad(texture<TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy); \
  __device__ float4 tex1DGrad(texture<u##TYPE##4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2DGrad(texture<T, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2DGrad(texture<TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float tex2DGrad(texture<unsigned TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float1 tex2DGrad(texture<TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float1 tex2DGrad(texture<u##TYPE##1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float2 tex2DGrad(texture<TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float2 tex2DGrad(texture<u##TYPE##2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float4 tex2DGrad(texture<TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy); \
  __device__ float4 tex2DGrad(texture<u##TYPE##4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex1DLayeredGrad(texture<T, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex1DLayeredGrad(texture<TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float tex1DLayeredGrad(texture<unsigned TYPE, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float1 tex1DLayeredGrad(texture<TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float1 tex1DLayeredGrad(texture<u##TYPE##1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float2 tex1DLayeredGrad(texture<TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float2 tex1DLayeredGrad(texture<u##TYPE##2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float4 tex1DLayeredGrad(texture<TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy); \
  __device__ float4 tex1DLayeredGrad(texture<u##TYPE##4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex2DLayeredGrad(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex2DLayeredGrad(texture<TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float tex2DLayeredGrad(texture<unsigned TYPE, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float1 tex2DLayeredGrad(texture<TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float1 tex2DLayeredGrad(texture<u##TYPE##1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float2 tex2DLayeredGrad(texture<TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float2 tex2DLayeredGrad(texture<u##TYPE##2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float4 tex2DLayeredGrad(texture<TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy); \
  __device__ float4 tex2DLayeredGrad(texture<u##TYPE##4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

template <class T>
__device__ T tex3DGrad(texture<T, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy);
#define MK_GETFLOAT(TYPE) \
  __device__ float tex3DGrad(texture<TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float tex3DGrad(texture<unsigned TYPE, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float1 tex3DGrad(texture<TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float1 tex3DGrad(texture<u##TYPE##1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float2 tex3DGrad(texture<TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float2 tex3DGrad(texture<u##TYPE##2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float4 tex3DGrad(texture<TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy); \
  __device__ float4 tex3DGrad(texture<u##TYPE##4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy);
MK_GETFLOAT(char)
MK_GETFLOAT(short)
#undef MK_GETFLOAT

#endif
