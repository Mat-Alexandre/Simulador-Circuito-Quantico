#include <stdio.h>
#include "quantum.cuh"

int main() {
	srand(time(NULL));
	// Tamanhos de estruturas
	int qbit_size = N * sizeof(qubit);
	int vect_size = T * sizeof(int);
	
	// Variáveis iniciais
	qubit q[N], *d_q;
	int mesure[N], *d_mesure;
	int teste[1], *d_teste;
	
	// Valores para teste
	q[0].amplitude[0].real = 1.0f;
	q[0].amplitude[0].imag = 0.0f;
	q[0].amplitude[1].real = 0.0f;
	q[0].amplitude[1].imag = 0.0f;

	// Vetores alvo e controles
	int t[T] = { 0, 1 }, c1[T] = { 2, 3 };
	int *d_t, *d_c1;

	// Alocação dos vetores Devices
	cudaMalloc((void**) &d_q, qbit_size);
	cudaMalloc((void**) &d_mesure, N*sizeof(int));

	// Cópia para as variáveis Devices
	cudaMemcpy( d_q,  q, qbit_size, cudaMemcpyHostToDevice);
	
	// Imprimindo os resultados iniciais
	printf("--AMPLITUDE INICIAL--\n");
	printQubit(q, NULL, N);
	
	// Aplicação da(s) porta(s)
	hadamardGate<<<B, N >>>(d_q);
	mesureQubit<<<B, N>>>(d_q, d_mesure, rand() % RAND_PRECISION);

	// Cópia para as variáveis Host
	cudaMemcpy( q, d_q,  qbit_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( mesure,   d_mesure, N*sizeof(int), cudaMemcpyDeviceToHost);
	
	// Imprimindo os resultados
	printf("---AMPLITUDE FINAL---\n");
	printQubit(q, mesure, N);
	
	// Liberando as variáveis alocadas
	cudaFree(d_q);
	cudaFree(d_mesure);

	return 0;
}
