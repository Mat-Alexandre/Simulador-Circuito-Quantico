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
	float real, imag;
} complex;

typedef struct qubit {
	complex amplitude[2]; // deeve ser alterado para [size]
	int size;
} qubit;

typedef struct simulator {
	qubit* q;
	int* mesure;
	int* target;
	int* control[2];
	int size;
} simulator;

/* QUBITS FUNCTIONS */

__host__ simulator initSimulatorDevice(int size);

__host__ simulator initSimulatorHost(int size);

__host__ void cpyToDevice(simulator ori, simulator dest);

__host__ void cpyToHost(simulator ori, simulator dest);

__host__ void freeSimulatorHost(simulator simu);

__host__ void freeSimulatorDevice(simulator d_simu);

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b);

__global__ void mesureQubit(qubit* q, int* mesure_vector, float percentage);

__host__ void printQubit(simulator sim, int mesure);

/* QUANTUM GATES */

__global__ void toffoliGate(qubit* d_q, int t, int* c1, int* c2);

__global__ void cnotGate(qubit* d_q, int* t, int* ctrl);

__global__ void notGate(qubit* d_q);

__global__ void hadamardGate(qubit* d_q);

__global__ void phaseGate(qubit* d_q);

__global__ void notGate_T(qubit* d_q, int target);

__global__ void hadamardGate_T(qubit* d_q, int target);

__global__ void phaseGate_T(qubit* q, int target);

#endif
