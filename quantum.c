#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct complex complex;
typedef struct qubit qubit;

struct complex{
    float real, imag;
};

struct qubit{
    complex *amplitude;
    int size;
};

/* QUBITS FUNCTIONS */

qubit initQubit(int size){
    qubit q;
    q.amplitude = (complex*)malloc(size * sizeof(complex));
    q.size = size;
    return q;
}

void freeQubit(qubit q){
    free(q.amplitude);
}

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b){
    complex c;
    c.real = (a.real*b.real) - (a.imag*b.imag);
    c.imag = (a.real*b.imag) + (a.imag*b.real);

    return c;
}

qubit tensorProduct(qubit q1, qubit q2){
    qubit res = initQubit(q1.size*q2.size);
    complex x;
    
    for(int i = 0; i < q1.size; i++){
        for(int j = 0; j < q2.size; j++){
            res.amplitude[q1.size * i + j] = complexProduct(q1.amplitude[i], q2.amplitude[j]);
        }
    }
    
    return res;
}

void printQubit(qubit q){
    for (int i = 0; i < q.size; i++) printf("(%.2f + %.2fi) * |%d>\n", q.amplitude[0].real, q.amplitude[0].imag, i);
}

/* QUANTUM GATES */

void notGate(qubit q){
    complex aux = q.amplitude[0];
    q.amplitude[0] = q.amplitude[1];
    q.amplitude[1] = aux;
}

void hadamardGate(qubit q){
    float ampH = 1/sqrt(2);
    complex alpha, beta;

    alpha.real = (q.amplitude[0].real + q.amplitude[1].real)*ampH;
    alpha.imag = (q.amplitude[0].real + q.amplitude[1].real)*ampH;

    beta.real = (q.amplitude[0].real - q.amplitude[1].real)*ampH;
    beta.imag = (q.amplitude[0].real - q.amplitude[1].real)*ampH;

    q.amplitude[0] = alpha;
    q.amplitude[1] = beta;
}

void phaseGate(qubit q){
   float b = -q.amplitude[1].imag;
   float c = q.amplitude[1].real;
   q.amplitude[1].real = b;
   q.amplitude[1].imag = c;
}

/* MAIN */

void main(int argc, char** argv){
    qubit q1 = initQubit(2);

    q1.amplitude[0].real = .25f;
    q1.amplitude[0].imag = .25f;
    q1.amplitude[1].real = .5f;
    q1.amplitude[1].imag = .5f;

    // hadamardGate(q1);
    notGate(q1);

    printQubit(q1);

    freeQubit(q1);
}