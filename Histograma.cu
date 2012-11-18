#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGHT 256
#define NUMBER_OF_THREADS 512

__global__ void GPUfuncion(*hist, *data, max)
{
	int t = threadIdx.x;
	int b = blockIdx.x;
	int B = blockDim.x;
	int value;

	__shared__ int hist_temp[HIST_LENGHT];

	int index = b * B + t;
	
	if (t < HIST_LENGHT)
		hist_temp[t] = 0;

	__syncthreads();

	if (index < max)
	{
		value = data[index];
		atomicAdd(hist_temp[value], 1);
	}

	__syncthreads();

}

int main()
{
	/**Guarda hora de inicio**/
	struct timeval before , after;
	gettimeofday(&before , NULL);
	
	int matrix_dim, array_lenght, *data_h, hist_h[HIST_LENGHT];
	int i;

	/** Ficheros de entrada y salida **/
	FILE *in_f = fopen("entrada", "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &matrix_dim);
	array_lenght = matrix_dim * matrix_dim;

	data_h = (int *)malloc(array_lenght * sizeof(int));
	for (i = 0; i < array_lenght && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);

	CUDA_Hist(data_h, hist_h, array_lenght);

	for (i = 0; i < 256; i++)
	{
		if (i == 255)
			fprintf(out, "%d", hist_h[i]);
		else
			fprintf(out, "%d\n", hist_h[i]);
	}
	
	fclose(in_f);
	fclose(out_f);
	
	/**Parar el reloj**/
	gettimeofday(&after , NULL);
	printf("Tiempo de ejecucion: %.0lf [ms]\n" , time_diff(before , after) );

	return 0;
}

void CUDA_Hist(int *data_h, int *hist_h, int array_lenght)
{
	int *data_d, *hist_d, blocks;
	int block_size = NUMBER_OF_THREADS;

	cudaMalloc((void **) &data_d, array_lenght * sizeof(int));
	cudaMalloc((void **) &hist_d, HIST_LENGHT * sizeof(int));

	cudaMemcpy(data_d, data_h, array_lenght * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(hist_d, 0, HIST_LENGHT * sizeof(int));

	blocks = ceil((float)array_lenght/block_size);

	GPUfuncion <<<blocks, block_size>>> (hist_d, data_d, array_lenght);

	cudaMemcpy(hist_h, hist_d, HIST_LENGHT * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(data_d);
	cudaFree(hist_d);
	return;
}

double time_diff(struct timeval x , struct timeval y)
{
	double x_ms , y_ms , diff;

	x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;

	diff = (double)y_ms - (double)x_ms;

	return diff;
}