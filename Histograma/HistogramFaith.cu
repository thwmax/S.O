#include <stdio.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define THREADS_P_BLOCK 512

int main(int argc, char *argv[])
{
	float elapsed_Time;
	int i, data_length, *data_h;
	unsigned int hist_h[HIST_LENGTH];
	int *data_d, *hist_d, blocks;
	cudaEvent_t start, stop;

	/** Timers **/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/** Input de datos **/
	FILE *in_f = fopen(argv[1], "r");
	fscanf(in_f, "%d", &data_length);
	data_length *= data_length;

	/** Declaracion dinamica del tamano del arreglo dependiendo de la matriz **/
	data_h = (int *)malloc(data_length * sizeof(int));
	for (i = 0; i < data_length && fscanf(in_f, "%d", &data_h[i]) == 1; ++i);
	fclose(in_f);

	/** Alloc para la memoria en GPU **/
	cudaMalloc((void **) &data_d, data_length * sizeof(int));
	cudaMemcpy(data_d, data_h, data_length * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &hist_d, HIST_LENGTH * sizeof(int));
	cudaMemset(hist_d, 0, HIST_LENGTH * sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	blocks = prop.multiProcessorCount;
	//histogram_kernel<<blocks*2, 256>>(data_d, data_length, hist_d);

	//cudaMemcpy(hist_h, hist_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(data_d);
	cudaFree(hist_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_Time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Tiempo de ejecucion: %3.3f ms\n", elapsed_Time);
	return 0;
}
