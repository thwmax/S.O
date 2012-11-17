all:
	gcc Histograma.c -o histogramaC.out

run:
	./histogramaC.out

clean:
	rm -rf *.out *~ salida

