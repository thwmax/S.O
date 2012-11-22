#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define PI 3.1415

void gauss (double sigma, double gauss_matrix[][5]);
void gpuComputing(double gauss_matrix[][5], int** image_matrix, int** final_matrix, int height, int width);

__global__ void kernel(int* image, int* final, double* gauss, int pitch, int pitch_i, int pitch_f, int height, int width)
{
	int i, j, abs_Pos, result;
	int x, y;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int fila = blockIdx.x + 2;

	if (tid >= width - 4)
		return;
	else
	{
		result = 0;
		abs_Pos = fila * width + (threadIdx.x +2);
		x = abs_Pos % width;
		
		for (int r = 0; r < 5; ++r) {
			int aux = r - 2;
        	float* gauss_row = (float*)((char*)gauss + r * pitch);
        	float* image_row = (float*)((char*)image + (fila + aux) * pitch_i));
        	
        	for (int c = 0; c < 5; ++c)
        	{
        		int aux2 = c - 2;
            	float gauss_element = gauss_row[c];
            	float image_element = image_row[x + aux2];
            	result += (gauss_element * image_element)/273;
        	}
        	float* final_row = (float*)((char*)final + fila * pitch_f);
        	final_row[x] = result;
        }
        return;
    }	
}


int main(int argc, char *argv[])
{
	int width, height, i, j;
	int **image_matrix, **final_matrix;
	int **auxiliar_matrix;
	double gauss_matrix[5][5];
	double sigma = strtod(argv[2], NULL);

	int *temp, *temp2, *temp3;

	FILE *in_f = fopen(argv[1], "r");
	//FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &width);
	fscanf(in_f, "%d", &height);

	image_matrix = (int**)malloc(width * sizeof(int*));
	temp = (int*)malloc(width * height * sizeof(int));
	for(i = 0; i < width; i++)
		image_matrix[i] = temp + (i * height);

	auxiliar_matrix = (int**)malloc((width + 4) * sizeof(int*));
	temp2 = (int*)malloc((width + 4) * (height + 4) * sizeof(int));
	for(i = 0; i < (width + 4); i++)
		auxiliar_matrix[i] = temp2 + (i * (height + 4));

	final_matrix = (int**)malloc((width + 4) * sizeof(int*));
	temp3 = (int*)malloc((width + 4) * (height + 4) * sizeof(int));
	for(i = 0; i < (width + 4); i++)
		final_matrix[i] = temp3 + (i * (height + 4));

	for (i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
			fscanf(in_f, "%d", &image_matrix[i][j]);
	}
	fclose(in_f);
	gauss(sigma, gauss_matrix);

	/** Mover la matriz **/
	for(i = 2; i < width + 2; i++)
	{
		for(j = 2; j < height + 2; j++){
			auxiliar_matrix[i][j] = image_matrix[i-2][j-2];
		}
	}

	/** Copiando filas **/
	for(i = 2; i < width + 2; i++)
	{
		auxiliar_matrix[i][0] = auxiliar_matrix[i][4];
		auxiliar_matrix[i][1] = auxiliar_matrix[i][3];
		auxiliar_matrix[i][height + 2] = auxiliar_matrix[i][height];
		auxiliar_matrix[i][height + 3] = auxiliar_matrix[i][height-1];
	}

	/** Copiando columnas **/
	for(i = 2; i < height + 2; i++)
	{
		auxiliar_matrix[0][i] = auxiliar_matrix[4][i];
		auxiliar_matrix[1][i] = auxiliar_matrix[3][i];
		auxiliar_matrix[width + 2][i] = auxiliar_matrix[width][i];
		auxiliar_matrix[width + 3][i] = auxiliar_matrix[width-1][i];
	}

	gpuComputing(gauss_matrix, auxiliar_matrix, final_matrix, height, width);
	free(temp);
	free(temp2);
	free(temp3);
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
			gauss_matrix[i][j] = gauss_matrix[i][j]*273;
			
	return;
}


void gpuComputing(double gauss_matrix[][5], int** image_matrix, int** final_matrix, int height, int width)
{
	int *d_image, *d_final;
	double *d_gauss;
	int blocks, threads;
	int dimension = height * width;

	size_t pitch, pitch_i, pitch_f;

	cudaMallocPitch(&d_gauss, &pitch, 5 * sizeof(double), 5);
	cudaMallocPitch(&d_image, &pitch_i, (width + 4) * sizeof(int), height + 4);
	cudaMallocPitch(&d_final, &pitch_f, width * sizeof(int), height);

	cudaMemcpy2D(d_gauss, pitch, gauss_matrix, 5 * sizeof(double), 5 * sizeof(double), 5, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_image, pitch_i, *image_matrix, (width + 4) * sizeof(int), (width + 4) * sizeof(int), height + 4, cudaMemcpyHostToDevice);

	kernel<<<1, 1>>>(d_image, d_final, d_gauss, pitch, pitch_i, pitch_f, (height + 4), (width + 4);

	cudaMemcpy2D(*final_matrix, width*sizeof(int), d_final, pitch_f, width*sizeof(int), height, cudaMemcpyDeviceToHost);
	cudaFree(d_final);
	cudaFree(d_image);
	cudaFree(d_gauss);

	return;
}