#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double time_diff(struct timeval x , struct timeval y);

int main(int argc, char *argv[])
{
	/**Guarda hora de inicio**/
	struct timeval before , after;
	
	
	int matrixsize, i, j, histogram[256], *numbers;
	
	/** Ficheros de entrada y salida **/
	FILE *in = fopen(argv[1], "r");
	FILE *out = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(in, "%d", &matrixsize);
	
	/**Llena el arreglo contador con ceros**/
	for (i = 0; i < 256; i++)
		histogram[i] = 0;
	
	/**Asigna dinamicamente el tamano necesario para almacenar los enteros**/
	numbers = (int *)malloc(matrixsize * matrixsize * sizeof(int));
	
	/**Asigna todos los enteros a un arreglo**/
	for (i = 0; i < matrixsize * matrixsize && fscanf(in, "%d", &numbers[i]) == 1; ++i);

	gettimeofday(&before , NULL);
	/**Recorre el arreglo y compara cada numero con la histogram para sumar 1 al contador del numero calzado**/
	for (i = 0; i < matrixsize * matrixsize; i++)
		histogram[numbers[i]]++;
	gettimeofday(&after , NULL);
	
	/**Escribe en el archivo out la cantidad de cada numero encontrado**/
	for (i = 0; i < 256; i++)
	{
		if (i == 255)
			fprintf(out, "%d", histogram[i]);
		else
			fprintf(out, "%d\n", histogram[i]);
	}
	
	fclose(in);
	fclose(out);
	
	/**Parar el reloj**/
	
	printf("Tiempo de ejecucion: %lf [ms]\n" , time_diff(before , after) );

	return 0;
}

double time_diff(struct timeval x , struct timeval y)
{
		double x_ms , y_ms , diff;
	   
		x_ms = (double)x.tv_sec*(double)1000 + (double)x.tv_usec/(double)1000;
		y_ms = (double)y.tv_sec*(double)1000 + (double)y.tv_usec/(double)1000;
	   
		diff = (double)y_ms - (double)x_ms;
	   
		return diff;
}