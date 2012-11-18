#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define NUMBER_OF_THREADS 512

/** Maneja la memoria del dispositivo y el huesped, ademas
 ** de la transferencia de datos entre estos dos **/
void CUDA_Hist(int *data_h, int *hist_h, int array_length);

/** Kernel CUDA, realiza la suma resultante en un histograma **/
__global__ void GPUfuncion(int *hist, int *data, int max)
{
	int t = threadIdx.x;
	int b = blockIdx.x;
	int B = blockDim.x;
	int buffer;

	/** Acceso a memoria compartida es 100x mas rapida que
	 ** memoria global, por lo tanto se crea un arreglo local al
	 ** bloque **/
	__shared__ int hist_temp[HIST_LENGTH];
	if (t < HIST_LENGTH)
	{
		hist_temp[t] = 0;
	}
	/** Barrera para asegurar integridad del arreglo compartido **/
	__syncthreads();

	int index = b * B + t;
	
	if (index < max)
	{
		/** Rescata el valor de la memoria global para cada hebra
		 ** de ejecucion **/
		buffer = data[index];
		atomicAdd(&(hist_temp[buffer]), 1);
		__syncthreads();
		/** Traspasa el resultado a la memoria global **/
		if (t < HIST_LENGTH)
			atomicAdd(&(hist[t]), hist_temp[t]);
	}
	else
		return;
}

int main(int argc, char *argv[])
{	
	float elapsedTime;
	cudaEvent_t start, stop;

	/** Comienza a registrar el tiempo de ejecucion **/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int matrix_dim, array_length, *data_h, hist_h[HIST_LENGTH];
	int i;

	/** Ficheros de entrada y salida **/
	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &matrix_dim);
	array_length = matrix_dim * matrix_dim;

	/** Se declara arreglo dinamico que contiene a la matriz serializada **/
	data_h = (int *)malloc(array_length * sizeof(int));
	for (i = 0; i < array_length && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);

	CUDA_Hist(data_h, hist_h, array_length);

	/** Imprime en la salida estandar el histograma resultante **/
	for (i = 0; i < 256; i++)
	{
		if (i == 255)
			fprintf(out_f, "%d", hist_h[i]);
		else
			fprintf(out_f, "%d\n", hist_h[i]);
	}
	
	fclose(in_f);
	fclose(out_f);
	
	/** Mediante CUDA API events se calcula el tiempo de ejecucion **/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Tiempo de ejecucion: %f [ms]\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);	
	return 0;
}

void CUDA_Hist(int *data_h, int *hist_h, int array_length)
{
	int *data_d, *hist_d, blocks;
	int block_size = NUMBER_OF_THREADS;

	cudaMalloc((void **) &data_d, array_length * sizeof(int));
	cudaMalloc((void **) &hist_d, HIST_LENGTH * sizeof(int));

	cudaMemcpy(data_d, data_h, array_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(hist_d, 0, HIST_LENGTH * sizeof(int));

	blocks = ceil((float)array_length/block_size);

	GPUfuncion <<<blocks, block_size>>> (hist_d, data_d, array_length);

	cudaMemcpy(hist_h, hist_d, HIST_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(data_d);
	cudaFree(hist_d);
	return;
}