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

template<class T, enum cudaTextureReadMode V>
T __device__ tex1D (texture<T,cudaTextureType1D,V>, float x);
template<class T, enum cudaTextureReadMode V>
T __device__ tex2D (texture<T,cudaTextureType2D,V>, float x, float y);
template<class T, enum cudaTextureReadMode V>
T __device__ tex3D (texture<T,cudaTextureType3D,V>, float x, float y, float z);

template<class T, enum cudaTextureReadMode V>
T __device__ tex1Dfetch (texture<T,cudaTextureType1D,V>, int x);
template<class T, enum cudaTextureReadMode V>
T __device__ tex2Dfetch (texture<T,cudaTextureType2D,V>, int x);
template<class T, enum cudaTextureReadMode V>
T __device__ tex3Dfetch (texture<T,cudaTextureType3D,V>, int x);

template<class T, enum cudaTextureReadMode V>
T __device__ texCubemap (texture<T,cudaTextureTypeCubemap,V>, float, float, float);

template<class T, enum cudaTextureReadMode V>
T __device__ texCubemapLayered (texture<T,cudaTextureTypeCubemapLayered,V>, float, float, float, int);

template<class T, enum cudaTextureReadMode V>
T __device__ tex1DLayered (texture<T,cudaTextureType1DLayered,V>, float, int);
template<class T, enum cudaTextureReadMode V>
T __device__ tex2DLayered (texture<T,cudaTextureType2DLayered,V>, float, float, int);

template<class T, enum cudaTextureReadMode V>
T __device__ tex1DLod (texture<T,cudaTextureType1D,V>, float, float, float);
template<class T, enum cudaTextureReadMode V>
T __device__ tex2DLod (texture<T,cudaTextureType2D,V>, float, float, float);
template<class T, enum cudaTextureReadMode V>
T __device__ tex3DLod (texture<T,cudaTextureType3D,V>, float, float, float);

template<class T>
static __device__ void tex1Dfetch(T *retVal, cudaTextureObject_t texObject, int x);
template<class T>
static __device__ void tex1D(T *retVal, cudaTextureObject_t texObject, float x);
template <class T>
static __device__ T tex1D(cudaTextureObject_t texObject, float x);

template <class T>
static __device__ void tex2D(T *retVal, cudaTextureObject_t texObject, float x, float y);
template <class T>
static __device__ T tex2D(cudaTextureObject_t texObject, float x, float y);

template <class T>
static __device__ void tex3D(T *retVal, cudaTextureObject_t texObject, float x, float y, float z);
template <class T>
static __device__ T tex3D(cudaTextureObject_t texObject, float x, float y, float z);

template <class T>
static __device__ void tex1DLayered(T *retVal, cudaTextureObject_t texObject, float x, int layer);
template <class T>
static __device__ T tex1DLayered(cudaTextureObject_t texObject, float x, int layer);

template <class T>
static __device__ void tex2DLayered(T *retVal, cudaTextureObject_t texObject, float x, float y, int layer);
template <class T>
static __device__ T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer);

template <class T>
static __device__ void texCubemap(T *retVal, cudaTextureObject_t texObject, float x, float y, float z);
template <class T>
static __device__ T texCubemap(cudaTextureObject_t texObject, float x, float y, float z);

template <class T>
static __device__ void texCubemapLayered(T *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);
template <class T>
static __device__ T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer);

template <class T>
static __device__ void tex2Dgather(T *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);
template <class T>
static __device__ T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0);

template <class T>
static __device__ void tex1DLod(T *retVal, cudaTextureObject_t texObject, float x, float level);
template <class T>
static __device__ T tex1DLod(cudaTextureObject_t texObject, float x, float level);

template <class T>
static __device__ void tex2DLod(T *retVal, cudaTextureObject_t texObject, float x, float y, float level);
template <class T>
static __device__ T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level);

template <class T>
static __device__ void tex3DLod(T *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);
template <class T>
static __device__ T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level);

template <class T>
static __device__ void tex1DLayeredLod(T *retVal, cudaTextureObject_t texObject, float x, int layer, float level);
template <class T>
static __device__ T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level);

template <class T>
static __device__ void tex2DLayeredLod(T *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);
template <class T>
static __device__ T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level);

template <class T>
static __device__ void texCubemapLod(T *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);
template <class T>
static __device__ T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level);

template <class T>
static __device__ void texCubemapLayeredLod(T *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);
template <class T>
static __device__ T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

template <class T>
static __device__ void tex1DGrad(T *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);
template <class T>
static __device__ T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

template <class T>
static __device__ void tex2DGrad(T *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);
template <class T>
static __device__ T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

template <class T>
static __device__ void tex3DGrad(T *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);
template <class T>
static __device__ T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

template <class T>
static __device__ void tex1DLayeredGrad(T *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);
template <class T>
static __device__ T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

template <class T>
static __device__ void tex2DLayeredGrad(T *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);
template <class T>
static __device__ T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

#endif
