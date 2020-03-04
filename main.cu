#include <stdio.h>
#include <windows.h>
#include "quantum.cuh"

int main() {
	// Variáveis iniciais
	srand(time(NULL)* time(NULL));
	float rand_value = rand() % RAND_PRECISION;
	float percentage = (float)(rand_value) / RAND_PRECISION;
	int quantidade = 2;

	simulator simu = initSimulatorHost(quantidade);
	simulator d_simu = initSimulatorDevice(quantidade);

	printQubit(simu, 0); // Visualização dos estados iniciais

	// Copiando valores para variável no device
	cpyToDevice(simu, d_simu);
	int t[] = {1}, *d_target, c[] = {0}, *d_c;
	cudaMalloc((void **)&d_target, sizeof(int));
	cudaMalloc((void **)&d_c, sizeof(int));
	cudaMemcpy(d_target, t, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);

	// Execução do Algoritmo de Grover
	hadamardGate<<<1, quantidade>>>(d_simu.q);
	//
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	cnotGate<<<1, quantidade>>>(d_simu.q, d_c, d_target);
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	//
	hadamardGate<<<1, quantidade>>>(d_simu.q);
	//
	notGate<<<1, quantidade>>>(d_simu.q);
	//
	hadamardGate_T << <1, quantidade >> > (d_simu.q, 1);
	cnotGate << <1, quantidade >> > (d_simu.q, d_c, d_target);
	hadamardGate_T << <1, quantidade >> > (d_simu.q, 1);
	//
	notGate << <1, quantidade >> > (d_simu.q);
	//
	hadamardGate << <1, quantidade >> > (d_simu.q);
	// Medindo qbits
	printf("Percent: %f\n", percentage);
	mesureQubit<<<1, quantidade>>>(d_simu.q, d_simu.mesure, percentage);

	// Copiando valores para variável no kernel
	cpyToHost(d_simu, simu);

	// Exibindo resultados
	printQubit(simu, 1);
	/*
	Resultado esperado: 100% -- 10
	*/
	cudaFree(d_c);
	cudaFree(d_target);
	freeSimulatorDevice(d_simu);
	freeSimulatorHost(simu);
	
	return 0;
}

/*

// Exemplo obtido através do vídeo https://youtu.be/Uw6zEMSxKvg por daytonellwanger
	hadamardGate<<<1, quantidade>>>(d_simu.q);
	//
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	cnotGate<<<1, quantidade>>>(d_simu.q, d_target, d_c);
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	//
	hadamardGate<<<1, quantidade>>>(d_simu.q);
	//
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	cnotGate<<<1, quantidade>>>(d_simu.q, d_target, d_c);
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	//
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	notGate_T<<<1, quantidade>>>(d_simu.q, 0);
	cnotGate<<<1, quantidade>>>(d_simu.q, d_target, d_c);
	notGate_T<<<1, quantidade>>>(d_simu.q, 0);
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 1);
	//
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 0);
	notGate_T<<<1, quantidade>>>(d_simu.q, 1);
	// Alterando os alvos e controles
	//d_target[0] = 0;
	//d_c[0] = 1;
	cnotGate<<<1, quantidade>>>(d_simu.q, d_target, d_c);
	notGate_T<<<1, quantidade>>>(d_simu.q, 1);
	hadamardGate_T<<<1, quantidade>>>(d_simu.q, 0);
	//
	hadamardGate<<<1, quantidade>>>(d_simu.q);

	// Medindo qbits
	printf("Percent: %f\n", percentage);
	mesureQubit<<<1, quantidade>>>(d_simu.q, d_simu.mesure, percentage);
	
	*/
