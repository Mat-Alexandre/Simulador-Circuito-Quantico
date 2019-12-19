#include "quantum.cuh"

/* QUBITS FUNCTIONS */

simulator initSimulatorDevice(int size) {
	simulator d_sim;
	int qbit_size = sizeof(qubit) * size;
	int array_size = sizeof(int) * size;
	
	cudaMalloc((void**) &d_sim, sizeof(simulator));
	cudaMalloc((void**) &d_sim.size, sizeof(int));
	cudaMalloc((void**) &d_sim.q, qbit_size);
	cudaMalloc((void**) &d_sim.mesure, array_size);
	cudaMalloc((void**) &d_sim.target, array_size);
	for(int i = 0; i < 2; i++)
		cudaMalloc((void**) &d_sim.control[i], array_size);

	d_sim.size = size;
	printf("Device criado com sucesso.\n");
	return d_sim;
}

simulator initSimulator(int size) {
	simulator sim;
	sim.size = size;
	sim.q = (qubit*)calloc(size, sizeof(qubit));
	
	sim.mesure = (int*)calloc(size, sizeof(int));
	sim.target = (int*)calloc(size, sizeof(int));
	for(int i = 0; i < 2; i++)
		sim.control[i] = (int*)calloc(size, sizeof(int));

	
	printf("Host criado com sucesso.\n");
	return sim;
}

void cpyToDevice(simulator ori, simulator dest) {

	int qbit_size = sizeof(qubit) * ori.size;
	int array_size = sizeof(int) * ori.size;

	cudaMemcpy(dest.q, ori.q, qbit_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dest.mesure, ori.mesure, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dest.target, ori.target, array_size, cudaMemcpyHostToDevice);
	for (int i = 0; i < 2; i++)
		cudaMemcpy(dest.control[i], ori.control[i], array_size, cudaMemcpyHostToDevice);

	printf("Copiado para device.\n");
}

void cpyToHost(simulator ori, simulator dest) {

	int qbit_size = sizeof(qubit) * ori.size;
	int array_size = sizeof(int) * ori.size;

	cudaMemcpy(ori.q, dest.q, qbit_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ori.mesure, dest.mesure, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ori.target, dest.target, array_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2; i++)
		cudaMemcpy(ori.control[i], dest.control[i], array_size, cudaMemcpyDeviceToHost);

	printf("Copiado para host.\n");
}

void freeSimulatorDevice(simulator d_simu) {
	cudaFree(d_simu.q);
	cudaFree(d_simu.mesure);
	cudaFree(d_simu.target);
	for(int i = 0; i < 2; i++)
		cudaFree(d_simu.control[i]);
	cudaFree(d_simu.control);
	printf("Device liberado.\n");
}

void freeSimulatorHost(simulator simu) {
	free(simu.q);
	free(simu.mesure);
	free(simu.target);
	for(int i = 0; i < 2; i++)
		free(simu.control[i]);
	printf("Host liberado.\n");
}

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b) {
	complex c;
	c.real = (a.real * b.real) - (a.imag * b.imag);
	c.imag = (a.real * b.imag) + (a.imag * b.real);

	return c;
}

__host__ void printQubit(qubit* q, int *result, int size) {
	printf("Qubit\t\n\t|");
	for (int i = 0; i < 2; i++) printf("------|%d>------", i); //vetor

	for (int i = 0; i < size; i++) {
		printf("\n%d\t|", i); // qbit
		for (int j = 0; j < q[i].size; j++) {
			printf("(%.2f + %.2fi)\t", q[i].amplitude[j].real, q[i].amplitude[j].imag); // amplitudes
		}
		printf("\n\n");
	}
	if (result != NULL) {
		printf("------RESULTADO------\n");
		printf("Qubit\t--\tColap.\n");
		for (int i = 0; i < size; i++) printf("%d\t\t%d\n", i, result[i]);
	}
}

__global__ void mesureQubit(qubit* q, int* mesure_vector, float percentage) {
	int index = threadIdx.x;

	float sum = .0f, prev = .0f;
	int mesure_result = 0;
	for (int i = 0; i < q[index].size; i++) {
		float a = q[index].amplitude[1].real * q[index].amplitude[1].real;
		float b = q[index].amplitude[1].imag * q[index].amplitude[1].imag;
		sum += (a + b);
		if (prev <= percentage && percentage <= sum) {
			mesure_vector[index] = i;
		}
		prev = sum;
	}
}

/* QUANTUM GATES */

// Toffoli e cnot precisam representar estados emaranhados
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

	float a1 = d_q[index].amplitude[0].real;
	float a2 = d_q[index].amplitude[0].imag;
	float b1 = d_q[index].amplitude[1].real;
	float b2 = d_q[index].amplitude[1].imag;

	alpha.real = (a1 + b1) * ampH;
	alpha.imag = (a2 + b2) * ampH;

	beta.real = (a1 - b1) * ampH;
	beta.imag = (a2 - b2) * ampH;

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

// As funções Multi aplicam o resultado da operação nos qbits indicados em target

__global__ void notGateMulti(qubit* d_q, int *target) {
	int index = threadIdx.x;
	complex aux = d_q[target[index]].amplitude[0];
	d_q[target[index]].amplitude[0] = d_q[target[index]].amplitude[1];
	d_q[target[index]].amplitude[1] = aux;
}

__global__ void hadamardGateMulti(qubit* d_q, int *target) {
	int index = threadIdx.x;

	float ampH = 0.70710678118;
	complex alpha, beta;

	float a1 = d_q[target[index]].amplitude[0].real;
	float a2 = d_q[target[index]].amplitude[0].imag;
	float b1 = d_q[target[index]].amplitude[1].real;
	float b2 = d_q[target[index]].amplitude[1].imag;

	alpha.real = (a1 + b1) * ampH;
	alpha.imag = (a2 + b2) * ampH;

	beta.real = (a1 - b1) * ampH;
	beta.imag = (a2 - b2) * ampH;

	d_q[target[index]].amplitude[0] = alpha;
	d_q[target[index]].amplitude[1] = beta;
}

__global__ void phaseGateMulti(qubit* q, int *target) {
	int index = threadIdx.x;
	float b = -q[target[index]].amplitude[1].imag;
	float c = q[target[index]].amplitude[1].real;
	q[target[index]].amplitude[1].real = b;
	q[target[index]].amplitude[1].imag = c;
}
