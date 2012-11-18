#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define HIST_LENGHT 256
#define NUMBER_OF_THREADS 512


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
