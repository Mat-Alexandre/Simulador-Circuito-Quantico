#ifndef QUANTUM_CUH_
#define QUANTUM_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// typedef struct qubit qubit;
// typedef struct compelx complex;

struct complex{
    float real, imag;
};

struct qubit{
    struct complex *amplitude;
    int size;
};

/* QUBITS FUNCTIONS */

struct qubit initQubit(int);

void freeQubit(struct qubit);

/* OTHER FUNCTIONS*/

struct complex complexProduct(struct complex, struct complex);

struct qubit tensorProduct(struct qubit, struct qubit);

void printQubit(struct qubit);

/* QUANTUM GATES */

void notGate(struct qubit);

void hadamardGate(struct qubit);

void phaseGate(struct qubit);

#endif