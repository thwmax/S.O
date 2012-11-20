#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define THREADS_P_BLOCK 512

void CUDA_Hist(int *data_h, int *hist_h, int array_length);

/** Funcion kernel de CUDA que procesa los datos **/
__global__ void GPUfuncion(int *hist, int *data, int max)
{
	__shared__ int shared_Hist[256];
	/** Declaracon de variables identificadoras de hebras **/
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int t = threadIdx.x;
	
	if ( i < array_length)
		int value = data[i];
	else
		return;

	if (t < 256)
		shared_Hist[t] = 0;
	__syncthread();

	atomicAdd(shared_Hist[value], 1);
	__syncthread();

	hist[t] += shared_Hist[t];
	return;
}

int main(int argc, char *argv[])
{
	/** Declaracion de variables para medir el tiempo **/
	cudaEvent_t start, stop;
	float elapsedTime;
	int i, array_length, *data_h, hist_h[HIST_LENGTH];
	
	/** Ficheros de entrada y salida **/
	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &array_length)
	array_length *= array_length;
	
	/** Declaracion dinamica del tamano del arreglo dependiendo de la matriz **/
	data_h = (int *)malloc(array_length * sizeof(int));
	for (i = 0; i < array_length && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);
	
	/** Llamado a la funcion de CUDA que procesa los datos **/
	CUDA_Hist(data_h, hist_h, array_length);
	
	/** Se imprime en el archivo **/
	for (i = 0; i < HIST_LENGTH; i++)
	{
		if (i == HIST_LENGTH - 1)
			fprintf(out_f, "%d", hist_h[i]);
		else
			fprintf(out_f, "%d\n", hist_h[i]);
	}
	
	fclose(in_f);
	fclose(out_f);
	return 0;
}

/** Funcion que recibe el puntero al arreglo con los datos, otro al histograma
    y el largo del arreglo, hace el copiado de memoria necesario y manda los datos
    a la funcion kernel para su procesamiento **/
void CUDA_Hist(int *data_h, int *hist_h, int array_length)
{
	int *data_d, *hist_d, blocks;
	
	/** Se reserva la memoria dinamicamente dependiendo el tamano de la matriz **/
	cudaMalloc((void **) &data_d, array_length * sizeof(int));
	cudaMalloc((void **) &hist_d, HIST_LENGTH * sizeof(int));
	
	/** Se copian los datos desde la memoria principal a la de la tarjeta grafica **/
	cudaMemcpy(data_d, data_h, array_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(hist_d, 0, HIST_LENGTH * sizeof(int));
	
	/** Se definen la cantidad de bloques necesarios para el paralelismo optimo **/
	blocks = (int)ceil(array_length/512.0);
	
	/** Llamado a la funcion kernel que procesa los datos **/
	GPUfuncion <<<blocks, THREADS_P_BLOCK>>> (hist_d, data_d, array_length);
	
	/** Se libera la memoria solicitada **/
	cudaFree(data_d);
	cudaFree(hist_d);
	return;
}