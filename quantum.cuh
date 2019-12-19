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

typedef struct simulator{
	qubit *q;
	int *mesure;
	int *target;
	int *control[2];
	int size;
} simulator;

/* QUBITS FUNCTIONS */

simulator initSimulatorDevice(int size);

simulator initSimulator(int size);

void cpyToDevice(simulator ori, simulator dest);

void cpyToHost(simulator ori, simulator dest);

void freeSimulatorHost(simulator simu);

void freeSimulatorDevice(simulator d_simu);

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

__global__ void notGateMulti(qubit* d_q, int *target);

__global__ void hadamardGateMulti(qubit* d_q, int *target);

__global__ void phaseGateMulti(qubit* q, int *target);

#endif
