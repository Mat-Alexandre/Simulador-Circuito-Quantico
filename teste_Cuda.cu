#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[threadIdx.x] = 2 * x[i] + y[i];
}

int main(void)
{
  int N = 20;
  float *x; 
  float *y;

  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));

  printf("oi");
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&x, N * sizeof(float));
  cudaMalloc(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  for (int i = 0; i < N; i++)
    printf("%f\n", y[i]);

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++)
    printf("%f\n", y[i]);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}