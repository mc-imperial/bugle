#ifndef CUDA_CURAND_H
#define CUDA_CURAND_H

__device__ unsigned int __bugle_random_uint();
__device__ unsigned long long __bugle_random_ull();
__device__ uint4 __bugle_random_uint4();
__device__ float __bugle_random_float();
__device__ double __bugle_random_double();
__device__ float2 __bugle_random_float2();
__device__ float4 __bugle_random_float4();
__device__ double2 __bugle_random_double2();

struct curandStateXORWOW {
  volatile unsigned int fake_state;
};
typedef struct curandStateXORWOW curandStateXORWOW_t;
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;

struct curandStateMRG32k3a {
  volatile unsigned int fake_state;
};
typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;

struct curandStateMtgp32 {
  volatile unsigned int fake_state[256];
};
typedef struct curandStateMtgp32 curandStateMtgp32_t;

struct curandStatePhilox4_32_10 {
  volatile unsigned int fake_state;
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

struct curandDiscreteDistribution_st {
  volatile unsigned int fake_state;
};
typedef struct curandDiscreteDistribution_st *curandDiscreteDistribution_t;

struct curandStateSobol32 {
  volatile unsigned int fake_state;
};
typedef struct curandStateSobol32 curandStateSobol32_t;

struct curandStateScrambledSobol32 {
  volatile unsigned int fake_state;
};
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;

struct curandStateSobol64 {
  volatile unsigned long long fake_state;
};
typedef struct curandStateSobol64 curandStateSobol64_t;

struct curandStateScrambledSobol64 {
  volatile unsigned long long fake_state;
};
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;


template <class T>
__device__ static __inline__ void
curand_init (
    unsigned long long seed, unsigned long long sequence,
    unsigned long long offset, T *state)
{
  state->fake_state = __bugle_random_uint();
}

template <class T>
__device__ static __inline__ unsigned int
curand (T *state) {
  state->fake_state += __bugle_random_uint();
  return __bugle_random_uint();
}

// Block specific restrictions are not checked
__device__ static __inline__ unsigned int
curand (curandStateMtgp32_t *state) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
  unsigned char index = (blockDim.z * blockDim.y * threadIdx.z)
                        + (blockDim.x * threadIdx.y)
                        + threadIdx.x;
#pragma GCC diagnostic pop
  state->fake_state[index] += __bugle_random_uint();
  return __bugle_random_uint();
}

// Block specific restrictions are not checked
// Restrictions on indexes are not checked
__device__ static __inline__ unsigned int
curandmtgp32specific (curandStateMtgp32_t *state, unsigned char index,
                      unsigned char n)
{
  state->fake_state[index] += __bugle_random_uint();
  return __bugle_random_uint();
}

/* Distributions */

template <class T>
__device__ static __inline__ float
curand_uniform (T *state)
{
  curand(state);
  return __bugle_random_float();
}

template <class T>
__device__ static __inline__ float
curand_normal (T *state)
{
  curand(state);
  return __bugle_random_float();
}

template <class T>
__device__ static __inline__ float
curand_log_normal (T *state, float mean, float stddev)
{
  curand(state);
  return __bugle_random_float();
}

template <class T>
__device__ static __inline__ double
curand_uniform_double (T *state)
{
  curand(state);
  return __bugle_random_double();
}

template <class T>
__device__ static __inline__ double
curand_normal_double (T *state)
{
  curand(state);
  return __bugle_random_double();
}

template <class T>
__device__ static __inline__ double
curand_log_normal_double (T *state, double mean, double stddev)
{
  curand(state);
  return __bugle_random_double();
}

template <class T>
__device__ static __inline__ unsigned int
curand_poisson (T *state, double lambda)
{
  curand(state);
  return __bugle_random_uint();
}


template <class T>
__device__ static __inline__ unsigned int
curand_discrete (T * state, curandDiscreteDistribution_t discrete_distribution)
{
  curand(state);
  discrete_distribution->fake_state += __bugle_random_uint();
  return __bugle_random_uint();
}

template <class T>
__device__ static __inline__ float2
curand_normal2 (T *state)
{
  curand(state);
  return __bugle_random_float2();
}

template <class T>
__device__ static __inline__ float2
curand_log_normal2 (T *state)
{
  curand(state);
  return __bugle_random_float2();
}

template <class T>
__device__ static __inline__ double2
curand_normal2_double (T *state)
{
  curand(state);
  return __bugle_random_double2();
}

template <class T>
__device__ static __inline__ double2
curand_log_normal2_double (T *state)
{
  curand(state);
  return __bugle_random_double2();
}

/* CUDA 5.5 */

__device__ static __inline__ uint4
curand4 (curandStatePhilox4_32_10_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_uint4();
}

__device__ static __inline__ float4
curand_uniform4 (curandStatePhilox4_32_10_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_float4();
}

__device__ static __inline__ float4
curand_normal4 (curandStatePhilox4_32_10_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_float4();
}

__device__ static __inline__ float4
curand_log_normal4 (curandStatePhilox4_32_10_t *state, float mean,
                    float stddev)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_float4();
}

__device__ static __inline__ uint4
curand_poisson4 (curandStatePhilox4_32_10_t *state, double lambda)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_uint4();
}

__device__ static __inline__ uint4
curand_discrete4 (curandStatePhilox4_32_10_t *state,
                  curandDiscreteDistribution_t discrete_distribution)
{
  state->fake_state += __bugle_random_uint();
  discrete_distribution->fake_state += __bugle_random_uint();
  return __bugle_random_uint4();
}

__device__ static __inline__ double2
curand_uniform2_double (curandStatePhilox4_32_10_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_double2();
}

__device__ static __inline__ double2
curand_normal2_double (curandStatePhilox4_32_10_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_double2();
}

__device__ static __inline__ double2
curand_log_normal2_double (curandStatePhilox4_32_10_t *state, double mean,
                           double stddev)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_double2();
}

/* Quasirandom Sequence Sobol */

template <class T>
__device__ static __inline__ void
curand_init (unsigned int *direction_vectors, unsigned int offset, T *state)
{
  state->fake_state = __bugle_random_uint();
}

__device__ static __inline__ void
curand_init (unsigned int *direction_vectors, unsigned int scramble_c,
             unsigned int offset, curandStateScrambledSobol32_t *state)
{
  state->fake_state = __bugle_random_uint();
}

__device__ static __inline__ void
curand_init (unsigned int *direction_vectors, unsigned int scramble_c,
             unsigned int offset, curandStateScrambledSobol64_t *state)
{
  state->fake_state = __bugle_random_uint();
}

__device__ static __inline__ unsigned long long
curand (curandStateSobol64_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_ull();
}

__device__ static __inline__ unsigned long long
curand (curandStateScrambledSobol64_t *state)
{
  state->fake_state += __bugle_random_uint();
  return __bugle_random_ull();
}

/* Skip ahead */

template <class T>
__device__ static __inline__ void
skipahead (unsigned long long n, T *state)
{
  curand(state);
}

__device__ static __inline__ void
skipahead (unsigned int n, curandStateSobol32_t *state)
{
  state->fake_state += __bugle_random_uint();
}

__device__ static __inline__ void
skipahead (unsigned int n, curandStateScrambledSobol32_t *state)
{
  state->fake_state += __bugle_random_uint();
}

template <class T>
__device__ static __inline__ void
skipaheadsequence (unsigned long long n, T *state)
{
  curand(state);
}

#endif
