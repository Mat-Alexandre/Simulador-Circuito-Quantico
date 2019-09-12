#include<stdio.h>
#include<stdlib.h>

#define N 8
// Kernel function to add the elements of two arrays
__global__
void add(float *z, float *x, float *y)
{
  int indice = blockIdx.x;
  if(indice < N)
    z[indice] = 2 * x[indice] + y[indice];
}

int main(void)
{
  float *x; 
  float *y;
  float *z;
  float *d_x;
  float *d_y;
  float *d_z;

  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  z = (float *) malloc(N * sizeof(float));
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

  printf("oi");
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc((void **) &d_x, N * sizeof(float));
  cudaMalloc((void **) &d_y, N * sizeof(float));
  cudaMalloc((void **) &d_z, N * sizeof(float));



  for (int i = 0; i < N; i++)
    printf("%f\n", y[i]);


  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  // Run kernel on 1M elements on the GPU
  
  add<<<N,1>>>(d_z, d_x, d_y);
  //cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();

  for (int i = 0; i < N; i++)
    printf("%f\n", z[i]);

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  return 0;
}