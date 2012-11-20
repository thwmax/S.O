#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define PI 3.1415

void gauss (double sigma, double gauss_matrix[][5]);
void gpuComputing(double gauss_matrix[][5], int* image_matrix, int* final_matrix, int height, int width);

__global__ void kernel(int* image, int* final, double* gauss, int pitch, int height, int width)
{
	final[0] = 9;
	return;
}

int main(int argc, char *argv[])
{
	int width, height;
	int i;
	int *image_matrix, *final_matrix;
	double gauss_matrix[5][5];
	double sigma = strtod(argv[2], NULL);

	FILE *in_f = fopen(argv[1], "r");
	//FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &width);
	fscanf(in_f, "%d", &height);

	image_matrix = (int *)malloc(width * height * sizeof(int*));
	final_matrix = (int *)malloc(width * height * sizeof(int*));

	for (i = 0; i < width * height; i++)
		fscanf(in_f, "%d", &image_matrix[i]);

	fclose(in_f);

	gauss(sigma, gauss_matrix);
	gpuComputing(gauss_matrix, image_matrix, final_matrix, height, width);

	printf("%d\n", final_matrix[0]);
	free(image_matrix);
	free(final_matrix);

	//fclose(out_f);

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

void gpuComputing(double gauss_matrix[][5], int* image_matrix, int* final_matrix, int height, int width)
{
	int *d_image, *d_final;
	double *d_gauss;
	
	size_t pitch;

	cudaMallocPitch((void** )&d_gauss, &pitch, 5 * sizeof(double), 5);
	cudaMalloc((void** )&d_image, width * height * sizeof(int));
	cudaMalloc((void** )&d_final, width * height * sizeof(int));

	cudaMemcpy2D(d_gauss, pitch, gauss_matrix, 5*sizeof(double), 5*sizeof(double), 5, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image_matrix, width * height * sizeof(int), cudaMemcpyHostToDevice); 

	kernel<<<1, 512>>>(d_image, d_final, d_gauss, pitch, height, width);

	cudaMemcpy(final_matrix, d_final, width*height*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_final);
	cudaFree(d_image);
	cudaFree(d_gauss);

	return;
}