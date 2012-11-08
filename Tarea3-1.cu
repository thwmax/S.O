#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

int main()
{
	int **array_gpu;
	int **array_cpu;
	int matrix_lenght;

	FILE *entrada = fopen("entrada", "r");
	fscanf(entrada, "%d", &matrix_lenght);

	cudaMalloc3D((void **) &array_gpu, array_lenght * sizeof(int));
	
	array_cpu = (int **) malloc(matrix_lenght * sizeof(int));
	*array_cpu = (int *) malloc(matrix_lenght * sizeof(int));

	
	return 0;
}
