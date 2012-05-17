typedef __attribute__((ext_vector_type(4))) int int4;

int4 mul(int4 x, int4 y, int4 z) {
  return x * y + z;
}
