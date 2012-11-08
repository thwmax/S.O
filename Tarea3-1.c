#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double time_diff(struct timeval x , struct timeval y);

/** comentario freak, lista[256] es lo mismo que decir tamanomatriz**/
int main(){
	/**Dar comienzo al reloj**/
	struct timeval before , after;
    	gettimeofday(&before , NULL);
	
	/** CODIGO REFERENTE A LAS MATRICES **/
	int tamanomatriz, i, j, lista[256], *numeros;
	FILE *entrada = fopen("entrada", "r");
	FILE *salida = fopen("salida", "w");
	
	/** Leer el primer numero que determina el tamano de la matriz **/
	fscanf(entrada, "%d", &tamanomatriz);
	
	/**Llena el arreglo contador con ceros**/
	for (i = 0; i < 256; i++)
		lista[i] = 0;
	/**Asigna dinamicamente el tamano necesario para almacenar los enteros**/
	numeros = (int *)malloc(tamanomatriz * tamanomatriz * sizeof(int));
	
	/**Asigna todos los enteros a un arreglo**/
	for (i = 0; i < tamanomatriz * tamanomatriz && fscanf(entrada, "%d", &numeros[i]) == 1; ++i);
	
	/**Recorre el arreglo y compara cada numero con la lista para sumar 1 al contador del numero calzado**/
	for (i = 0; i < tamanomatriz * tamanomatriz; i++)
		lista[numeros[i]]++;
	
	/**Escribe en el archivo salida la cantidad de cada numero encontrado**/
	for (i = 0; i < 256; i++){
		if (i == 255)
			fprintf(salida, "%d", lista[i]);
		else
			fprintf(salida, "%d\n", lista[i]);
	}
	
	fclose(entrada);
	fclose(salida);
	
	/**Parar el reloj**/
	gettimeofday(&after , NULL);
     	printf("Tiempo de ejecucion: %.0lf [ms]\n" , time_diff(before , after) );

	return 0;
}

double time_diff(struct timeval x , struct timeval y)
{
        double x_ms , y_ms , diff;
       
        x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
        y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
       
        diff = (double)y_ms - (double)x_ms;
       
        return diff;
}
