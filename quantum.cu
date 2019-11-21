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

__host__ void printQubit(qubit* q, int *result, int size) {
	printf("Qubit\treal\timag\n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < 2; j++) {
			printf("%d.%d\t%.2f\t%.2f\n", i, j, q[i].amplitude[j].real, q[i].amplitude[j].imag);
		}
		printf("\n");
	}
	if (result != NULL) {
		printf("------RESULTADO------\n");
		printf("Qubit\t--\tValor\n");
		for (int i = 0; i < size; i++) printf("%d\t\t%d\n", i, result[i]);
	}
}

__global__ void mesureQubit(qubit *q, int *mesure_vector, int rand_value) {
	int index = threadIdx.x;

	int v[N][RAND_PRECISION];
	// O cálculo da amplitude deve ser verificado
	float amp = (q[index].amplitude[0].real )*(q[index].amplitude[0].real)*100.0F;

	int fraction = (int)amp;
	
	for (int i = 0; i < fraction; i++)
		v[index][i] = 0;
	for (int i = fraction; i < RAND_PRECISION; i++)
		v[index][i] = 1;
	
	mesure_vector[index] = v[index][rand_value];
}

/* QUANTUM GATES */

__global__ void toffoliGate(qubit* d_q, int* t, int* c1, int* c2) {
	int index = threadIdx.x;

	// A aplcação da porta só pode ser efetuada se os vetores possuirem o mesmo tamanho
	if ((sizeof(t) / sizeof(t[0])) == (sizeof(c1) / sizeof(c1[0])) &&
		(sizeof(t) / sizeof(t[0])) == (sizeof(c2) / sizeof(c2[0])))
	// Se os qubit c1 e c2 possuirem amplitudes do vetor |1> diferente de 0, trocar o sinal do qubit em t
	if ((d_q[c1[index]].amplitude[1].real != .0f || d_q[c1[index]].amplitude[1].imag != .0f) &&
		(d_q[c2[index]].amplitude[1].real != .0f || d_q[c2[index]].amplitude[1].imag != .0f)) {
		complex aux = d_q[t[index]].amplitude[0];
		d_q[t[index]].amplitude[0] = d_q[t[index]].amplitude[1];
		d_q[t[index]].amplitude[1] = aux;
	}
}

__global__ void cnotGate(qubit* d_q, int* t, int* ctrl) {
	// t é um ponteiro para vetor de qubits a serem afetados pela porta cnotGate
	// ctrl é um ponteiro para vetor de qubits de controle
	int index = threadIdx.x;
	// Se o qubit ctrl possuir amplitude do vetor |1> diferente de 0, trocar o sinal do qubit em t
	if((sizeof(t) / sizeof(t[0])) == (sizeof(ctrl) / sizeof(ctrl[0])))
	if (d_q[ctrl[index]].amplitude[1].real != .0f || d_q[ctrl[index]].amplitude[1].imag != .0f) {
		complex aux = d_q[t[index]].amplitude[0];
		d_q[t[index]].amplitude[0] = d_q[t[index]].amplitude[1];
		d_q[t[index]].amplitude[1] = aux;
	}
}

__global__ void notGate(qubit* d_q) {
	int index = threadIdx.x;

	complex aux = d_q[index].amplitude[0];
	d_q[index].amplitude[0] = d_q[index].amplitude[1];
	d_q[index].amplitude[1] = aux;
}

__global__ void hadamardGate(qubit* d_q) {
	int index = threadIdx.x;

	float ampH = 0.70710678118;
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
