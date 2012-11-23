#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define PI 3.1415

void gauss (int sigma, int gauss_matrix[][5]);
void gpuComputing(int gauss_matrix[][5], int** image_matrix, int** final_matrix, int height, int width);

__global__ void kernel(int* image, int* final, int* gauss, int pitch, int pitch_i, int pitch_f, int height, int width)
{
	int abs_Pos, c, r;
	int x, aux, aux2;
	int gauss_element;
	int image_element;
	int tid = threadIdx.x;

	float result;

	/** Todos los threads de un bloque comparten la misma fila **/
	__shared__ int fila;

	fila = blockIdx.x + 2;

	/** Los threads van recorriendo la fila cada 512 casilleros **/
	while(tid < width - 4)
	{
		result = 0;

		/** Posicion absoluta del thread en el arreglo con la imagen **/
		abs_Pos = fila * width + (tid +2);

		/** Posicion x en la matriz **/
		x = abs_Pos % width;

		/** Recorre la matriz de gauss y simultaneamente copia los valores necesarios de
		 ** la matriz de imagen para realizar el calculo **/
		for (r = 0; r < 5; ++r) {
			aux = r - 2;
			int* gauss_row = (int*)((char*)gauss + r * pitch);
			int* image_row = (int*)((char*)image + ((fila + aux) * pitch));
			for (c = 0; c < 5; ++c)
			{
				aux2 = c - 2;
				gauss_element = gauss_row[c];
				image_element = image_row[x + aux2];
				result += (gauss_element * image_element)/273;
				
			}
		}
		
		/** Guarda el valor a la celda correspondiente en la imagen final **/
		int* final_row = (int*)((char*)final +  blockIdx.x * pitch_f);
		final_row[tid] = result;
		
		/** Incremento para realizar todas las operaciones de una fila en un bloque **/
		tid += 512;
	}	
	return;
}


int main(int argc, char *argv[])
{
	int width, height, i, j;
	int **image_matrix, **final_matrix, **auxiliar_matrix;
	int gauss_matrix[5][5];
	int sigma = (int)strtod(argv[2], NULL);
	int *temp, *temp2, *temp3;

	/** Lectura del archivo contenedor de los datos **/
	FILE *in_f = fopen(argv[1], "r");
	
	/** Lee el ancho de la matriz **/
	fscanf(in_f, "%d", &width);
	/** Lee el alto de la matriz **/
	fscanf(in_f, "%d", &height);

	/** Arreglo dinamico 2D para almacenar la matriz **/
	image_matrix = (int**)malloc(width * sizeof(int*));
	temp = (int*)malloc(width * height * sizeof(int));
	for(i = 0; i < width; i++)
		image_matrix[i] = temp + (i * height);

	/** Arrelgo dinamico 2D con 4 filas y columnas adicionales **/
	auxiliar_matrix = (int**)malloc((width + 4) * sizeof(int*));
	temp2 = (int*)malloc((width + 4) * (height + 4) * sizeof(int));
	for(i = 0; i < (width + 4); i++)
		auxiliar_matrix[i] = temp2 + (i * (height + 4));

	/** Arreglo dinamico 2D para almacenar el resultado final **/
	final_matrix = (int**)malloc(width * sizeof(int*));
	temp3 = (int*)malloc(width * height * sizeof(int));
	for(i = 0; i < width; i++)
		final_matrix[i] = temp3 + (i * height );

	/** Lee los valores del archivo y los almacena en image_matrix **/
	for (i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
			fscanf(in_f, "%d", &image_matrix[i][j]);
	}
	fclose(in_f);

	/** Calculo de la matriz con los valores del filtro **/
	gauss(sigma, gauss_matrix);
	
	/** Desplazamiento de la matriz 2 filas hacia arriba y 2 hacia la izquierda **/
	for(i = 2; i < width + 2; i++)
	{
		for(j = 2; j < height + 2; j++){
			auxiliar_matrix[i][j] = image_matrix[i-2][j-2];
		}
	}

	/** Reflexion de filas **/
	for(i = 2; i < width + 2; i++)
	{
		auxiliar_matrix[i][0] = auxiliar_matrix[i][4];
		auxiliar_matrix[i][1] = auxiliar_matrix[i][3];
		auxiliar_matrix[i][height + 2] = auxiliar_matrix[i][height];
		auxiliar_matrix[i][height + 3] = auxiliar_matrix[i][height-1];
	}

	/** Reflexion de columnas **/
	for(i = 0; i < height + 4; i++)
	{
		auxiliar_matrix[0][i] = auxiliar_matrix[4][i];
		auxiliar_matrix[1][i] = auxiliar_matrix[3][i];
		auxiliar_matrix[width + 2][i] = auxiliar_matrix[width][i];
		auxiliar_matrix[width + 3][i] = auxiliar_matrix[width-1][i];
	}

	gpuComputing(gauss_matrix, auxiliar_matrix, final_matrix, height, width);
	
	/** Abre archivo que contendra los valores finales **/
	FILE *out = fopen("salida", "w");
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if ( j != width - 1)
				fprintf(out, "%d\t", final_matrix[i][j]);
			else if (i != height-1)
				fprintf(out, "%d\n", final_matrix[i][j]);
			else
				fprintf(out, "%d", final_matrix[i][j]);
		}
	}
	fclose(out);

	/** Liberacion de memoria correspondiente a las 3 matrices **/
	free(temp);
	free(temp2);
	free(temp3);

	return 0;
}

void gauss(int sigma, int gauss_matrix[][5])
{
	int i, j;
	int x = -2, y = 2;
	double u, v, varianza;
	double operation;

	for(i = 0; i < 5; i++)
	{
		for(j = 0; j < 5; j++)
		{
			u = pow(x,2);
			v = pow(y,2);
			varianza = pow(sigma,2);

			operation = 273.0*exp((-u-v)/(2.0*varianza))/(2.0*PI*varianza);
			gauss_matrix[i][j] = (int)ceil(operation);
			x++;
		}
		y--;
		x = -2;
	}
	return;
}


void gpuComputing(int gauss_matrix[][5], int** image_matrix, int** final_matrix, int height, int width)
{
	int *d_image, *d_final, *d_gauss;
	int r_height = height + 4;
	int r_width = width + 4;

	size_t pitch, pitch_i, pitch_f;

	/** Metodo que entrega la direccion del arreglo en el dispositivo **/
	cudaMallocPitch(&d_gauss, &pitch, 5 * sizeof(int), 5);
	cudaMallocPitch(&d_image, &pitch_i, r_width * sizeof(int), r_height);
	cudaMallocPitch(&d_final, &pitch_f, width * sizeof(int), height);
	
	/** Traspaso de los datos desde el host al dispositivo **/
	cudaMemcpy2D(d_gauss, pitch, *gauss_matrix, 5 * sizeof(int), 5 * sizeof(int), 5, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_image, pitch_i, *image_matrix, r_width * sizeof(int), r_width * sizeof(int), r_height, cudaMemcpyHostToDevice);

	/** Llamada al kernel, si bien su implementacion pudo ser mas eficiente, los tiempos permanecen bajos **/
	kernel<<<height, 512>>>(d_image, d_final, d_gauss, pitch, pitch_i, pitch_f, r_height, r_width);

	/** Traspaso del resultado final **/
	cudaMemcpy2D(*final_matrix, width*sizeof(int), d_final, pitch_f, width*sizeof(int), height, cudaMemcpyDeviceToHost);
	
	cudaFree(d_final);
	cudaFree(d_image);
	cudaFree(d_gauss);

	return;
}
