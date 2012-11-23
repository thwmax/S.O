#include <stdio.h>
#include <cuda.h>

#define HIST_LENGTH 256
#define THREADS_P_BLOCK 512

__global__ void histogram_kernel(int *data_d, int data_length, int *hist_d)
{
	int index, offset;
	/** Arreglo en memoria compartida **/
	__shared__ int temp_hist[HIST_LENGTH];

	/**Inicializacion en 0 **/
	temp_hist[threadIdx.x] = 0;
	__syncthreads();

	index = threadIdx.x + blockIdx.x * blockDim.x;

	/** Cada thread debe recorrer los datos segun el offset declarado **/
	offset = blockDim.x * gridDim.x;

	/** Llenado del histograma **/
	while(index < data_length) {
		atomicAdd( &temp_hist[data_d[index]], 1);
		index += offset;
	}

	__syncthreads();

	/** Traspaso a memoria global **/
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

	cudaEventRecord(start, 0);

	/** Alloc para la memoria en GPU **/
	cudaMalloc((void **) &data_d, data_length * sizeof(int));

	/** Se copian los datos del histograma a la memoria del dispositivo **/
	cudaMemcpy(data_d, data_h, data_length * sizeof(int), cudaMemcpyHostToDevice);

	/** Se reserva la memoria para el histograma en el dispositivo y se inicializa en 0 **/
	cudaMalloc((void **) &hist_d, HIST_LENGTH * sizeof(int));
	cudaMemset(hist_d, 0, HIST_LENGTH * sizeof(int));

	/** Se analizo que los tiempos de ejecucion eran menores al enviar dos veces el numero de
	 ** multiprocesadores presentes **/
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	blocks = prop.multiProcessorCount * 2;

	/** 256 threads es el numero optimo para la ejecucion **/
	histogram_kernel<<<blocks, 256>>>(data_d, data_length, hist_d);

	/** Se traspasan los datos del histograma desde el dispositivo **/
	cudaMemcpy(hist_h, hist_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	
	/** Se detiene el timer **/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_Time, start, stop);

	cudaFree(data_d);
	cudaFree(hist_d);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/** Datos escritos en el archivo de salida **/
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
