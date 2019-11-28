#ifndef _QUANTUM_CUH
#define _QUANTUM_CUH

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#define RAND_PRECISION 10000
#define N 1
#define T 2
#define B 1

typedef struct complex {
	float real = 1.0f, imag = .0f;
} complex;

typedef struct qubit {
	complex amplitude[2];
	int size = 2;
} qubit;

/* QUBITS FUNCTIONS */

qubit initQubit(int size);

void freeQubit(qubit q);

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b);

__global__ void mesureQubit(qubit* q, int* mesure_vector, float percentage);

__host__ void printQubit(qubit* q, int* result, int size);

/* QUANTUM GATES */

__global__ void toffoliGate(qubit* d_q, int* t, int* c1, int* c2);

__global__ void cnotGate(qubit* d_q, int* t, int* ctrl);

__global__ void notGate(qubit* d_q);

__global__ void hadamardGate(qubit* d_q);

__global__ void phaseGate(qubit* d_q);

__global__ void notGateRange(qubit* d_q, int a, int b);

__global__ void hadamardGateRange(qubit* d_q, int a, int b);

__global__ void phaseGateRange(qubit* q, int a, int b);

#endif
