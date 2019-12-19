#include <stdio.h>
#include <windows.h>
#include "quantum.cuh"

int main() {
	// Vari�veis iniciais
	srand(time(NULL));
	float rand_value = rand() % RAND_PRECISION;
	float percentage = (float)(rand_value) / RAND_PRECISION;
	int quantidade;

	printf("Quantidade de qbits> \n");
	scanf("%d", &quantidade);

	simulator simu = initSimulator(quantidade);
	simulator d_simu = initSimulatorDevice(quantidade);

	// Execução
	cpyToDevice(simu, d_simu);
	
	hadamardGate<<<1, quantidade>>>(d_simu.q);
	mesureQubit<<<1, quantidade>>> (d_simu.q , d_simu.mesure, percentage);
	
	cpyToHost(d_simu, simu);

	printQubit(simu.q, simu.mesure, simu.size);

	freeSimulatorDevice(d_simu);
	freeSimulatorHost(simu);
	return 0;
}
