typedef __attribute__((ext_vector_type(4))) float float4;

float4 mul(float4 x, float4 y, float4 z) {
  return x * y + z;
}
