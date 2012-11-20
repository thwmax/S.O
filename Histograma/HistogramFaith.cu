#include <stdio.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define THREADS_P_BLOCK 512

__global__ void histogram_kernel(int *data_d, int data_length, int *hist_d)
{
	int index, offset;
	__shared__ int temp_hist[256];

	temp_hist[threadIdx.x] = 0;
	__syncthreads();

	index = threadIdx.x + blockIdx.x * blockDim.x;
	offset = blockDim.x * gridDim.x;

	while(index < data_length) {
		atomicAdd( &temp_hist[data_d[index]], 1);
		index += offset;
	}

	__syncthreads();
	atomicAdd(&(hist_d[threadIdx.x]), temp_hist[threadIdx.x]);
}

int main(int argc, char *argv[])
{
	float elapsed_Time;
	int i, blocks;
	int data_length;
	int hist_h[HIST_LENGTH], *data_h, *data_d, *hist_d;
	cudaEvent_t start, stop;

	/** Timers **/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	
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
	blocks *= 2;
	
	cudaEventRecord(start, 0);
	
	histogram_kernel<<<blocks, 256>>>(data_d, data_length, hist_d);
	cudaMemcpy(hist_h, hist_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_Time, start, stop);

	cudaFree(data_d);
	cudaFree(hist_d);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	FILE *out = fopen("salida", "w");
	for (i = 0; i < 256; i++)
	{
		if (i == 255)
			fprintf(out, "%d", hist_h[i]);
		else
			fprintf(out, "%d\n", hist_h[i]);
	}
	fclose(out);
	
	printf("Tiempo de ejecucion: %f ms\n", elapsed_Time);
	return 0;
}
