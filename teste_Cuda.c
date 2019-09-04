#include<stdio.h>
#include<stdlib.h>

//Função do Kernel pra adicionar elementos a 2 arrays 
__global__ 
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
}