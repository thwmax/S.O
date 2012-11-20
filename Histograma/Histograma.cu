#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define NUMBER_OF_THREADS 512

void CUDA_Hist(int *data_h, int *hist_h, int array_length);

/** Funcion kernel de CUDA que procesa los datos **/
__global__ void GPUfuncion(int *hist, int *data, int max)
{
	/** Declaracon de variables identificadoras de hebras **/
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int t = threadIdx.x;

	/** Si la hebra esta en el rango del histograma, se declara un auxiliar
	    con la posicion y se suma 1 en el arreglo del histograma **/
	if (i < max) {
		int aux = data[i];
		atomicAdd(&hist[aux], 1);
	}
}

int main(int argc, char *argv[])
{
	/** Declaracion de variables para medir el tiempo **/
	cudaEvent_t start, stop;
	float elapsedTime;
	int i, matrix_dim, array_length, *data_h, hist_h[HIST_LENGTH];
	
	/** Ficheros de entrada y salida **/
	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &matrix_dim);
	array_length = matrix_dim * matrix_dim;
	
	/** Declaracion dinamica del tamano del arreglo dependiendo de la matriz **/
	data_h = (int *)malloc(array_length * sizeof(int));
	for (i = 0; i < array_length && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);
	
	/** Se da inicio al reloj **/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/** Llamado a la funcion de CUDA que procesa los datos **/
	CUDA_Hist(data_h, hist_h, array_length);
	
	/** Se detiene el reloj **/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	/** Se calcula la diferencia entre ambos tiempos **/
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	/** Se imprime en el archivo **/
	for (i = 0; i < HIST_LENGTH; i++)
	{
		if (i == HIST_LENGTH - 1)
			fprintf(out_f, "%d", hist_h[i]);
		else
			fprintf(out_f, "%d\n", hist_h[i]);
	}
	
	printf("Tiempo de ejecucion: %f [ms]\n", elapsedTime);
	
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
	int block_size = NUMBER_OF_THREADS;
	
	/** Se reserva la memoria dinamicamente dependiendo el tamano de la matriz **/
	cudaMalloc((void **) &data_d, array_length * sizeof(int));
	cudaMalloc((void **) &hist_d, HIST_LENGTH * sizeof(int));
	
	/** Se copian los datos desde la memoria principal a la de la tarjeta grafica **/
	cudaMemcpy(data_d, data_h, array_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(hist_d, 0, HIST_LENGTH * sizeof(int));
	
	cudaMemcpy(hist_d, hist_h, HIST_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
	
	/** Se definen la cantidad de bloques necesarios para el paralelismo optimo **/
	blocks = (int)ceil(array_length/512.0);
	
	/** Llamado a la funcion kernel que procesa los datos **/
	GPUfuncion <<<blocks, block_size, HIST_LENGTH * sizeof(int)>>> (hist_d, data_d, array_length);
	
	/** Se libera la memoria solicitada **/
	cudaFree(data_d);
	cudaFree(hist_d);
	return;
}