#include "quantum.cuh"

/* QUBITS FUNCTIONS */

qubit initQubit(int size) {
	qubit q;
	q.amplitude[0].imag = .0f;
	q.amplitude[0].real = .0f;
	q.size = size;
	return q;
}

void freeQubit(qubit q) {
	free(q.amplitude);
}

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b) {
	complex c;
	c.real = (a.real * b.real) - (a.imag * b.imag);
	c.imag = (a.real * b.imag) + (a.imag * b.real);

	return c;
}

qubit tensorProduct(qubit q1, qubit q2) {
	qubit res = initQubit(q1.size * q2.size);

	for (int i = 0; i < q1.size; i++) {
		for (int j = 0; j < q2.size; j++) {
			res.amplitude[q1.size * i + j] = complexProduct(q1.amplitude[i], q2.amplitude[j]);
		}
	}

	return res;
}

void printQubit(qubit q) {
	for (int i = 0; i < q.size; i++) printf("(%.2f + %.2fi) * |%d>\n", q.amplitude[0].real, q.amplitude[0].imag, i);
}

/* QUANTUM GATES */

__global__ void notGate(qubit* d_q) {
	int index = threadIdx.x;

	complex aux = d_q[index].amplitude[0];
	d_q[index].amplitude[0] = d_q[index].amplitude[1];
	d_q[index].amplitude[1] = aux;
}

__global__ void hadamardGate(qubit* d_q) {
	int index = threadIdx.x;

	float ampH = 0.70710678118; //1/sqrt(2) : sqrt não pode ser utilizado por ser uma função host
	complex alpha, beta;

	alpha.real = (d_q[index].amplitude[0].real + d_q[index].amplitude[1].real) * ampH;
	alpha.imag = (d_q[index].amplitude[0].real + d_q[index].amplitude[1].real) * ampH;

	beta.real = (d_q[index].amplitude[0].real - d_q[index].amplitude[1].real) * ampH;
	beta.imag = (d_q[index].amplitude[0].real - d_q[index].amplitude[1].real) * ampH;

	d_q[index].amplitude[0] = alpha;
	d_q[index].amplitude[1] = beta;
}

__global__ void phaseGate(qubit* d_q) {	
	int index = threadIdx.x;
	float b = -d_q[index].amplitude[1].imag;
	float c = d_q[index].amplitude[1].real;
	d_q[index].amplitude[1].real = b;
	d_q[index].amplitude[1].imag = c;
}

__global__ void notGateRange(qubit* d_q, int a, int b) {
	// Executa a função nos bits dentro do range de a -- b
	int index = threadIdx.x;
	if (index >= a && index <= b) {
		complex aux = d_q[index].amplitude[0];
		d_q[index].amplitude[0] = d_q[index].amplitude[1];
		d_q[index].amplitude[1] = aux;
	}
}

__global__ void hadamardGateRange(qubit* d_q, int a, int b) {
	int index = threadIdx.x;

	if (index >= a && index <= b) {

		float ampH = 0.70710678118;
		complex alpha, beta;

		alpha.real = (d_q[index].amplitude[0].real + d_q[index].amplitude[1].real) * ampH;
		alpha.imag = (d_q[index].amplitude[0].real + d_q[index].amplitude[1].real) * ampH;

		beta.real = (d_q[index].amplitude[0].real - d_q[index].amplitude[1].real) * ampH;
		beta.imag = (d_q[index].amplitude[0].real - d_q[index].amplitude[1].real) * ampH;

		d_q[index].amplitude[0] = alpha;
		d_q[index].amplitude[1] = beta;
	}
}

__global__ void phaseGateRange(qubit* q, int a, int b) {
	int index = threadIdx.x;
	if (index >= a && index <= b) {
		float b = -q[index].amplitude[1].imag;
		float c = q[index].amplitude[1].real;
		q[index].amplitude[1].real = b;
		q[index].amplitude[1].imag = c;
	}
}