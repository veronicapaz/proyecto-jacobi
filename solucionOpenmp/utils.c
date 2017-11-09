
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif


double ** dmalloc_2d(int N) {
	if (N <= 0 ) return NULL;
	double **A = malloc(N * sizeof(double *));
	if (A == NULL) return NULL;
	A[0] = malloc(N*N*sizeof(double));
	if (A[0] == NULL) {
		free(A);
		return NULL;
	}
	int i;
	for (i = 1; i < N; i++)
		A[i] = A[0] + i * N;
	return A;
}

double * dmalloc_1d(int N){

	double *B;
	B = (double *) malloc(N*sizeof(double));

	return B;
}

void set_A(double** A, int N){
	int i;
	A[0][0] = -4;
	double n = sqrt( (double)N );
	double delta = 2.0 / (double)n;
	delta = delta * delta;
	for ( i = 1; i < N; i++)
	{
		A[i][i] = -4 /delta;
		A[i-1][i] = 1/delta;
		A[i][i-1] = 1/delta;
	}
}

void displayArray(double **A, int N)
{
	int i,j;
	for (j = 0; j < N; j++)
	{
		for (i=0; i<N;i++)
		{
			printf( "%f" , A[i][j]  );
			printf(" ");
		}
		printf("\n");
	}

}

void writeArray2file(double** A, int N)
{
	FILE* fp;
	fp = fopen("result.txt", "w");
	int i,j;
	for (j = 0; j < N; j++)
	{
		for (i=0; i<N;i++)
		{
			fprintf( fp,"%f," , A[i][j]  );
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void writeVec2file(double* x, int N)
{
	FILE* fp;
	fp = fopen("result_x.txt", "w");
	int i,j;
	int k = 0;
	int n = sqrt(N);
	for (j = 0; j < n; j++)
	{
		for (i=0; i<n;i++)
		{
			fprintf( fp,"%f," , x[k]  );
			k++;
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void set_B (double**B, int N){

	set_x0(B,N);

	double v = 200;
	int i,j;
	i = 0;
	double offset = 2.0 / (double)N;
	for (i = 0 ; i < N; i++)
	{
		double y = -1.0 + 2.0 * (float)i/(double)N;
		for (j = 0; j < N ; j++)
		{
			double x = -1.0 + 2.0 * (float)j/(double)N;
			if (x >= 0 && x <= 1.0/3.0 && y >= -2.0/3.0 && y <= -1.0/3.0)
				B[i][j] = 200.0;
			else
				B[i][j] = 0.0;
		}
	}
}

void set_x0(double**B, int N){

	double v = 200;
	int i,j;
	i = 0;
	for (i = 0 ; i < N; i++)
	{
		double y = -1.0 + 2.0 * (float)i/(double)N;
		for (j = 0; j < N ; j++)
		{
			B[i][j] = 0.0;
			if ( i == N-1 )
				B[i][j] = 20.0;
			if ( i == 0 )
				B[i][j] = 0.0;
			if ( j == 0 )
				B[i][j] = 20.0;
			if ( j == N-1 )
				B[i][j] = 20.0;
		}
	}
}

void displayVect(double *B, int N)
{
	int i;
	for(i=0; i<N; i++){
		printf("%f  ", B[i]);
	}
}


void free_A(double **A){
    free(A[0]);
    free(A);
}

void free_B(double *B){
	free(B);
}