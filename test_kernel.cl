__kernel void vector_add(__global const int *a, __global const int *b,
                         __global int *c) {
  int i = get_group_id(0);

  c[i] = b[i] + a[i];
}