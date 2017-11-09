#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "gaussseidel.h"
#include "jacobi.h"

int main(int argc, char *argv[]) {
	double threshold = 0.00001;
	int iter_max = 100000;
	int real_size = 64;

	// Looks if N is defined as argument.
	if (argc > 1) {
		printf("The argument supplied for N is: %s\n", argv[1]);
		real_size = atoi(argv[1]);
	} else {
		printf("Usage: ./poisson N=64 threshold=0 method=jacobi\n");
	}
	if (argc > 2) {
		printf("The argument supplied for threshold is: %s\n", argv[2]);
		threshold = strtod(argv[2], NULL);
	}

	int N = real_size+2;
	double **A = dmalloc_2d(N);
	double **b = dmalloc_2d(N);
	set_A(A,N);
	set_B(b,N);

	// Choose a method
	double **x;
	if (argc > 3) {
		printf("The argument supplied for the method is: %s\n", argv[3]);
		if (strcmp(argv[3],"jacobi") == 0) {
			printf("executing jacobi sequential...\n");
			x = jacobi(A,b,N, iter_max, threshold);
		} else if (strcmp(argv[3],"gaussseidel") == 0) {
			printf("executing gaussseidel sequential...\n");
			x = gaussseidel(A,b,N, iter_max, threshold);
		} else if (strcmp(argv[3],"parallel_jacobi") == 0) {
			printf("executing jacobi parallel...\n");
			x = parallel_jacobi(A,b,N, iter_max, threshold);
		} else if (strcmp(argv[3],"parallel2_jacobi") == 0) {
			printf("executing jacobi parallel more optimized...\n");
			x = parallel2_jacobi(A,b,N, iter_max, threshold);
		} else {
			printf("Unknown argument %s will execute gaussseidel sequential\n", argv[3]);
			x = gaussseidel(A,b,N, iter_max, threshold);
		}
	} else {
		printf("No argument suplied will use gaussseidel sequential with N: %i and threshold: %e\n", real_size, threshold);
		x = gaussseidel(A,b,N, iter_max, threshold);
	}

	// save result
	writeArray2file(x,N);

	return 0;
}