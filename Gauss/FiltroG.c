#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cuda.h>

#define PI 3.14

void gauss (double sigma, double gauss_matrix[][5]);

int main(int argc, char *argv[])
{
	int width, height;
	int i, j;
	int **image_matrix;
	double gauss_matrix[5][5];
	int *temp;
	char *finalPtr;
	double sigma = strtod(argv[2], &finalPtr);

	FILE *in_f = fopen(argv[1], "r");
	FILE *out_f = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in_f, "%d", &width);
	fscanf(in_f, "%d", &height);

	image_matrix = (int **)malloc(width * sizeof(int*));
	temp = (int *)malloc(width * height * sizeof(int));

	for (i = 0; i < width; i++)
		image_matrix[i] = temp + (i * height);

	for (i = 0; i < width; i++){
		for (j = 0; j < height; ++j){
			fscanf(in_f, "%d", &image_matrix[i][j]);
		}
	}

	gauss(sigma, gauss_matrix);

	printf("%f\n", gauss_matrix[2][2]);

	fclose(in_f);
	fclose(out_f);

	return 0;
}

void gauss(double sigma, double gauss_matrix[][5])
{
	int i, j;
	double x = -2.0, y = 2.0;
	double u, v, varianza;
	double sum = 0;

	for(i = 0; i < 5; i++)
	{
		for(j = 0; j < 5; j++)
		{
			u = pow(x,2);
			v = pow(y,2);
			varianza = pow(sigma,2);

			gauss_matrix[i][j] = exp((-u-v)/(2*varianza))/(2*PI*varianza);
			printf("(x,y) = (%f, %f) coef = %f \n", x,y, gauss_matrix[i][j]);
			sum += gauss_matrix[i][j];
			x++;
		}
		y--;
		x = -2.0;
	}

	for(i = 0; i < 5; i++)
		for(j = 0; j < 5; j++)
			gauss_matrix[i][j] *= sum;
			
	return;
}