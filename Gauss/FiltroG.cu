#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define PI 3.1415

void gauss (double sigma, double gauss_matrix[][5]);
void gpuComputing(double** gauss_matrix, int** image_matrix, int** final_matrix, int height, int width);

__global__ void kernel(int* image, int* final, double* gauss, int pitch, int pitch2, int pitch3, int height, int width)
{
	int i,j;
	int* row;

	for (i = 0; i < height; ++i)
	{
		row = (int*)((char*)final + i*pitch2);
		for(j = 0; j < width; ++j)
		{
			row[j] = 1;
		}
	}
}

int main(int argc, char *argv[])
{
	int width, height;
	int i, j;
	char *finalPtr;
	double sigma = strtod(argv[2], &finalPtr);

	int **image_matrix, **final_matrix;
	
	int *temp, *temp2;
	double gauss_matrix[5][5];

	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &width);
	fscanf(in_f, "%d", &height);

	image_matrix = (int **)malloc(width * sizeof(int*));
	final_matrix = (int **)malloc(width * sizeof(int*));

	temp = (int *)malloc(width * height * sizeof(int));
	temp2 = (int *)malloc(width * height * sizeof(int));

	for (i = 0; i < width; i++)
		image_matrix[i] = temp + (i * height);
	for (i = 0; i < width; i++)
		final_matrix[i] = temp2 + (i * height);

	for (i = 0; i < width; i++){
		for (j = 0; j < height; ++j){
			fscanf(in_f, "%d", &image_matrix[i][j]);
		}
	}

	gauss(sigma, gauss_matrix);

	printf("%d\n", final_matrix[0][0]);
	fclose(in_f);
	fclose(out_f);

	return 0;
}

void gauss(double sigma, double gauss_matrix[][5])
{
	int i, j;
	double x = -2.0, y = 2.0;
	double u, v, varianza;

	for(i = 0; i < 5; i++)
	{
		for(j = 0; j < 5; j++)
		{
			u = pow(x,2);
			v = pow(y,2);
			varianza = pow(sigma,2);

			gauss_matrix[i][j] = exp((-u-v)/(2*varianza))/(2*PI*varianza);
			x++;
		}
		y--;
		x = -2.0;
	}

	for(i = 0; i < 5; i++)
		for(j = 0; j < 5; j++)
			gauss_matrix[i][j] = ceil(gauss_matrix[i][j]*273);
			
	return;
}

void gpuComputing(double** gauss_matrix, int** image_matrix, int** final_matrix, int height, int width)
{
	int *d_image, *d_final;
	double *d_gauss;

	size_t pitch, pitch2, pitch3;

	cudaMallocPitch((void** )&d_gauss, &pitch, 5 * sizeof(double), 5);
	cudaMallocPitch((void** )&d_image, &pitch2, width*sizeof(int), height);
	cudaMallocPitch((void** )&d_final, &pitch3, width*sizeof(int), height);

	cudaMemcpy2D(d_gauss, pitch, gauss_matrix, 5*sizeof(double), 5*sizeof(double), 5, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_image, pitch2, image_matrix, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);

	kernel<<<1, 512>>>(d_image, d_final, d_gauss, pitch2, pitch3, pitch, height, width);

	cudaMemcpy2D(d_final, width*sizeof(int), final_matrix, pitch3, width*sizeof(int), height, cudaMemcpyDeviceToHost);
	cudaFree(d_final);
	cudaFree(d_image);
	cudaFree(d_gauss);
	return;

}