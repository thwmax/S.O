#include <stdio.h>
#include <stdlib.h>

int main(){

	int **i;

	i = (int **)malloc(3 * sizeof(int));
	*i = (int *) malloc(3 * sizeof(int));

	i[0][0] = 1;
	printf("Hola %d\n", i[0][0]);
	return 0;
} 
