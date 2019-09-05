#include "quantum.cuh"

/* QUBITS FUNCTIONS */

struct qubit initQubit(int size){
    struct qubit q;
    q.amplitude = (struct complex*)malloc(size * sizeof(struct complex));
    q.size = size;
    return q;
}

void freeQubit(struct qubit q){
    free(q.amplitude);
}

/* OTHER FUNCTIONS*/

struct complex complexProduct(struct complex a, struct complex b){
    struct complex c;
    c.real = (a.real*b.real) - (a.imag*b.imag);
    c.imag = (a.real*b.imag) + (a.imag*b.real);

    return c;
}

struct qubit tensorProduct(struct qubit q1, struct qubit q2){
    struct qubit res = initQubit(q1.size*q2.size);
    
    for(int i = 0; i < q1.size; i++){
        for(int j = 0; j < q2.size; j++){
            res.amplitude[q1.size * i + j] = complexProduct(q1.amplitude[i], q2.amplitude[j]);
        }
    }
    
    return res;
}

void printQubit(struct qubit q){
    for (int i = 0; i < q.size; i++) printf("(%.2f + %.2fi) * |%d>\n", q.amplitude[0].real, q.amplitude[0].imag, i);
}

/* QUANTUM GATES */

void notGate(struct qubit q){
    struct complex aux = q.amplitude[0];
    q.amplitude[0] = q.amplitude[1];
    q.amplitude[1] = aux;
}

void hadamardGate(struct qubit q){
    float ampH = 1/sqrt(2);
    struct complex alpha, beta;

    alpha.real = (q.amplitude[0].real + q.amplitude[1].real)*ampH;
    alpha.imag = (q.amplitude[0].real + q.amplitude[1].real)*ampH;

    beta.real = (q.amplitude[0].real - q.amplitude[1].real)*ampH;
    beta.imag = (q.amplitude[0].real - q.amplitude[1].real)*ampH;

    q.amplitude[0] = alpha;
    q.amplitude[1] = beta;
}

void phaseGate(struct qubit q){
   float b = -q.amplitude[1].imag;
   float c = q.amplitude[1].real;
   q.amplitude[1].real = b;
   q.amplitude[1].imag = c;
}