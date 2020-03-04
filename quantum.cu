#include "quantum.cuh"

/* QUBITS FUNCTIONS */

__host__ simulator initSimulatorDevice(int size) {
	simulator d_sim;
	int qbit_size = sizeof(qubit) * size;
	int array_size = sizeof(int) * size;

	cudaMalloc((void**)&d_sim, sizeof(simulator));
	cudaMalloc((void**)&d_sim.size, sizeof(int));
	cudaMalloc((void**)&d_sim.q, qbit_size);
	cudaMalloc((void**)&d_sim.mesure, array_size);
	cudaMalloc((void**)&d_sim.target, array_size);
	for (int i = 0; i < size; i++)
		cudaMalloc((void**)&d_sim.control[i], array_size);

	//d_sim.size = size;
	// printf("Device criado com sucesso.\n");
	return d_sim;
}

__host__ simulator initSimulatorHost(int size) {
	simulator sim;
	sim.size = size;
	sim.q = (qubit*)calloc(size, sizeof(qubit));
	if (sim.q == NULL) exit(-1);
	for (int i = 0; i < sim.size; i++)
		sim.q[i].amplitude[0].real = (float)1;

	sim.mesure = (int*)calloc(size, sizeof(int));
	if (sim.mesure == NULL) exit(-1);

	sim.target = (int*)calloc(size, sizeof(int));
	if (sim.target == NULL) exit(-1);

	for (int i = 0; i < size; i++) {
		sim.control[i] = (int*)calloc(size, sizeof(int));
		if (sim.control[i] == NULL) exit(-1);
	}

	// printf("Host criado com sucesso.\n");
	return sim;
}

__host__ void cpyToDevice(simulator ori, simulator dest) {

	int qbit_size = sizeof(qubit) * ori.size;
	int array_size = sizeof(int) * ori.size;

	dest.size = ori.size;
	cudaMemcpy(dest.q, ori.q, qbit_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dest.mesure, ori.mesure, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dest.target, ori.target, array_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dest.size, ori.size, sizeof(int), cudaMemcpyHostToDevice);
	for (int i = 0; i < ori.size; i++)
		cudaMemcpy(dest.control[i], ori.control[i], array_size, cudaMemcpyHostToDevice);

	// printf("Copiado para device.\n");
}

__host__ void cpyToHost(simulator ori, simulator dest) {

	int qbit_size = sizeof(qubit) * dest.size;
	int array_size = sizeof(int) * dest.size;

	cudaMemcpy(dest.q, ori.q, qbit_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dest.mesure, ori.mesure, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dest.target, ori.target, array_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < dest.size; i++)
		cudaMemcpy(dest.control[i], ori.control[i], array_size, cudaMemcpyDeviceToHost);

	// printf("Copiado para host.\n");
}

__host__ void freeSimulatorDevice(simulator d_simu) {
	cudaFree(d_simu.q);
	cudaFree(d_simu.mesure);
	cudaFree(d_simu.target);
	for (int i = 0; i < 2; i++)
		cudaFree(d_simu.control[i]);
	cudaFree(d_simu.control);
	// printf("Device liberado.\n");
}

__host__ void freeSimulatorHost(simulator simu) {
	free(simu.q);
	free(simu.mesure);
	free(simu.target);
	for (int i = 0; i < 2; i++)
		free(simu.control[i]);
	// printf("Host liberado.\n");
}

/* OTHER FUNCTIONS*/

complex complexProduct(complex a, complex b) {
	complex c;
	c.real = (a.real * b.real) - (a.imag * b.imag);
	c.imag = (a.real * b.imag) + (a.imag * b.real);

	return c;
}

__host__ void printQubit(simulator sim, int mesure) {
	for (int i = 0; i < sim.size; i++) {
		printf("Qubit [%d]:\n", i);
		for (int j = 0; j < 2; j++) {
			printf("|%d>: (%.2f + %.2fi)\t", j, sim.q[i].amplitude[j].real, sim.q[i].amplitude[j].imag);
		}
		printf("\n\n");
	}
	if (mesure)
		for (int i = 0; i < sim.size; i++) {
			printf("Qubit [%d]: %d\n", i, sim.mesure[i]);
		}
}

__global__ void mesureQubit(qubit* q, int* mesure_vector, float percentage) {
    // Input: qubit array, output array, random number
	// Output: collapsed values for each qubit according to percentage	int index = threadIdx.x;
	// A |0> + B |1>
    // |A|² + |B|² = 1
    // A = a + bi
	// A*A = (a + bi)(a + bi) = a² + 2abi - b²
	int index = threadIdx.x;
	
	float a = q[index].amplitude[0].real;
	float b = q[index].amplitude[0].imag;
	float ASquare = a*a + 2*a*b - b*b;

	if(ASquare >= percentage) mesure_vector[index] = 0;
	else mesure_vector[index] = 1;
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
	if ((sizeof(t) / sizeof(t[0])) == (sizeof(ctrl) / sizeof(ctrl[0])))
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

// As funções _T aplicam o resultado da operação no qbit indicado em target

__global__ void notGate_T(qubit* d_q, int target) {
	complex aux = d_q[target].amplitude[0];
	d_q[target].amplitude[0] = d_q[target].amplitude[1];
	d_q[target].amplitude[1] = aux;
}

__global__ void hadamardGate_T(qubit* d_q, int target) {
	float ampH = 0.70710678118;
	complex alpha, beta;

	float a1 = d_q[target].amplitude[0].real;
	float a2 = d_q[target].amplitude[0].imag;
	float b1 = d_q[target].amplitude[1].real;
	float b2 = d_q[target].amplitude[1].imag;

	alpha.real = (a1 + b1) * ampH;
	alpha.imag = (a2 + b2) * ampH;

	beta.real = (a1 - b1) * ampH;
	beta.imag = (a2 - b2) * ampH;

	d_q[target].amplitude[0] = alpha;
	d_q[target].amplitude[1] = beta;
}

__global__ void phaseGate_T(qubit* q, int target) {
	float b = -q[target].amplitude[1].imag;
	float c = q[target].amplitude[1].real;
	q[target].amplitude[1].real = b;
	q[target].amplitude[1].imag = c;
}
