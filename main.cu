#include <stdio.h>
#include <windows.h>
#include "quantum.cuh"

int main() {
	// Variáveis iniciais
	srand(time(NULL));
	float rand_value = rand() % RAND_PRECISION;
	float percentage = (float)(rand_value) / RAND_PRECISION;
	int quantidade = 2;

	simulator simu = initSimulatorHost(quantidade);
	simulator d_simu = initSimulatorDevice(quantidade);

	printQubit(simu, 0); // Visualização dos estados iniciais

	// Copiando valores para variável no device
	cpyToDevice(simu, d_simu);
	
	// Execução do Algoritmo de Grover
	// Exemplo obtido através do vídeo https://youtu.be/Uw6zEMSxKvg por daytonellwanger

	hadamardGate_All<<<1, quantidade>>>(d_simu);
	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);
	cnotGate<<<1, quantidade>>>(d_simu.q, {1}, {0});
	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);
	hadamardGate_All<<<1, quantidade>>>(d_simu);

	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);
	cnotGate<<<1, quantidade>>>(d_simu.q, {1}, {0});
	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);

	notGate<<<1, quantidade>>>(d_simu.q[0]);
	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);
	cnotGate<<<1, quantidade>>>(d_simu.q, {1}, {0});
	notGate<<<1, quantidade>>>(d_simu.q[0]);
	hadamardGate<<<1, quantidade>>>(d_simu.q[1]);

	hadamardGate<<<1, quantidade>>>(d_simu.q[0]);
	notGate<<<1, quantidade>>>(d_simu.q[1]);
	cnotGate<<<1, quantidade>>>(d_simu.q, {0}, {1});
	hadamardGate<<<1, quantidade>>>(d_simu.q[0]);
	notGate<<<1, quantidade>>>(d_simu.q[1]);

	hadamardGate_All<<<1, quantidade>>>(d_simu);


	// Medindo qbits
	mesureQubit<<<1, quantidade>>> (d_simu.q , d_simu.mesure, percentage);
	
	// Copiando valores para variável no kernel
	cpyToHost(d_simu, simu);

	// Exibindo resultados
	printQubit(simu, 1);
	/*
	Resultado esperado: 100% -- 10
	*/

	freeSimulatorDevice(d_simu);
	freeSimulatorHost(simu);
	return 0;
}
