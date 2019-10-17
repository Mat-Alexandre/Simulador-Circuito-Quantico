#ifndef _QUANTUM_CUH
#define _QUANTUM_CUH

#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

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

qubit tensorProduct(qubit q1, qubit q2);

void printQubit(qubit q);

/* QUANTUM GATES */

__global__ void notGate(qubit *d_q);

__global__ void hadamardGate(qubit *d_q);

__global__ void phaseGate(qubit *d_q);

__global__ void notGateRange(qubit* d_q, int a, int b);

__global__ void hadamardGateRange(qubit* d_q, int a, int b);

__global__ void phaseGateRange(qubit* q, int a, int b);

/*
cnot, toffoli em n qbits, emaranhamento, aplicação em um determinado qbit
*/

#endif