#include <stdio.h>
#include "quantum.cuh"

#define N 4
#define T 2

int main() {
	int qbit_size = N * sizeof(qubit);
	int vect_size = T * sizeof(int);
	
	qubit q[N], *d_q;
	
	// Valores para teste
	q[0].amplitude[1].real = 0.0f;
	q[1].amplitude[1].real = 0.0f;
	q[2].amplitude[0].real = 0.0f;
	q[3].amplitude[0].real = 0.0f;

	// Vetores alvo e controles
	int t[T] = { 0, 1 }, c1[T] = { 2, 3 }, c2[T];
	int *d_t, *d_c1, *d_c2;

	// Alocação dos vetores Devices
	cudaMalloc((void**) &d_q, qbit_size);
	cudaMalloc((void**) &d_t, vect_size);
	cudaMalloc((void**)&d_c1, vect_size);

	// Cópia para as variáveis Devices
	cudaMemcpy( d_q,  q, qbit_size, cudaMemcpyHostToDevice);
	cudaMemcpy( d_t,  t, vect_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c1, c1, vect_size, cudaMemcpyHostToDevice);
	
	// Imprimindo os resultados iniciais
	printf("-------INICIO-------\n");
	printQubit(q, N);
	printf("Qubit alvo: %d\n", t[0]);
	printf("Qubit controle: %d\n", c1[0]);

	// Aplicação da(s) porta(s)
	cnotGate<<<1, N>>>(d_q, d_t, d_c1);

	// Cópia para as variáveis Host
	cudaMemcpy( q, d_q,  qbit_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( t, d_t,  vect_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c1, d_c1, vect_size, cudaMemcpyDeviceToHost);

	// Imprimindo os resultados
	printf("------RESULTADO-----\n");
	printQubit(q, N);
	
	// Liberando as variáveis alocadas
	cudaFree(d_q);
	cudaFree(d_t);
	cudaFree(d_c1);

	return 0;
}
