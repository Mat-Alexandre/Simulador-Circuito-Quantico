#include <stdio.h>
#include "quantum.cuh"

__global__
void notGate(struct qubit *Q){
    
}

int main(){
    struct qubit q;
    struct qubit *d_q;

    // Allocating the qubit pointer
    cudaMalloc((void **) &d_q, sizeof(struct qubit));

    // Copying the data from q to d_q
    cudaMemcpy(d_q, &q, sizeof(struct qubit), cudaMemcpyHostToDevice);

    // Executing the notGate on GPU
    notGate<<<1,1>>>(d_q);

    // Copying back the value of d_q to q
    cudaMemcpy(&q, d_q, sizeof(struct qubit), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_q);

    return 0;
}