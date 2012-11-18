#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define NUMBER_OF_THREADS 512

void CUDA_Hist(int *data_h, int *hist_h, int array_length);

__global__ void GPUfuncion(int *hist, int *data, int max)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int t = threadIdx.x;

	__shared__ int local_hist[256];

	if (t < 256)
		local_hist[t] = 0;

	__syncthreads();

	if (i < max) {
		int aux = data[i];
		atomicAdd(&local_hist[aux], 1);
	}

	__syncthreads();

	if (t < 256)
		atomicAdd(&hist[t], local_hist[t]);
}

int main(int argc, char *argv[])
{	
	float elapsedTime;
	cudaEvent_t start, stop;

	int matrix_dim, array_length, *data_h, hist_h[HIST_LENGTH];
	int i;

	/** Ficheros de entrada y salida **/
	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &matrix_dim);
	array_length = matrix_dim * matrix_dim;

	data_h = (int *)malloc(array_length * sizeof(int));
	for (i = 0; i < array_length && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	CUDA_Hist(data_h, hist_h, array_length);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);	

	for (i = 0; i < 256; i++)
	{
		if (i == 255)
			fprintf(out_f, "%d", hist_h[i]);
		else
			fprintf(out_f, "%d\n", hist_h[i]);
	}
	
	fclose(in_f);
	fclose(out_f);
	
	printf("Tiempo de ejecucion: %f [ms]\n", elapsedTime);

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
	
	cudaMemcpy(hist_d, hist_h, HIST_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

	blocks = (int)ceil(array_length/512.0);

	GPUfuncion <<<blocks, block_size, 256 * sizeof(int)>>> (hist_d, data_d, array_length);

	cudaFree(data_d);
	cudaFree(hist_d);
	return;
}