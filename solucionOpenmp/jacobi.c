#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "jacobi.h"
#include "utils.h"

double** jacobi( double **A, double** b, int N, int nb_iter, double threshold)
{
	printf("INIT Jacobi (sequential)\n");
	double ** x = dmalloc_2d(N);
	double ** x_tmp = dmalloc_2d(N);
	// used for update
	double ** tmp = 0;
	set_x0(x, N);
	set_x0(x_tmp, N);
	int i, j ,k;
	k = 0;
	double d = INFINITY;
	double delta = 2.0 / ((double)(N-3));
	while ( d > threshold && k < nb_iter )
	{
		d = 0.0;
		for (i = 1; i < N-1; i++)
		{
			for (j = 1; j < N-1; j++)
			{
				x_tmp[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1] + delta * delta * b[i][j])/4.0;
				d += x_tmp[i][j]*x_tmp[i][j] - x[i][j]*x[i][j];
			}
		}
		tmp = x;
		x = x_tmp;
		x_tmp = tmp;
		k++;
	}
	printf("jacobi - k is: %i\n", k);
	free_A(x_tmp);
	return x;
}

double** parallel_jacobi( double **A, double** b, int N, int nb_iter, double threshold)
{
	printf("INIT Jacobi (parallel)\n");
	double ** x = dmalloc_2d(N);
	double ** x_tmp = dmalloc_2d(N);
	// used for update
	double ** tmp = 0;
	set_x0(x, N);
	set_x0(x_tmp, N);
	int i, j ,k;
	k = 0;
	double d = INFINITY;
	double delta = 2.0 / ((double)(N-3));
	while ( d > threshold && k < nb_iter )
	{
		d = 0.0;
		// variables are shared by default
		#pragma omp parallel for private(i,j) reduction(+:d)
		for (i = 1; i < N-1; i++)
		{
			for (j = 1; j < N-1; j++)
			{
				x_tmp[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1] + delta * delta * b[i][j])*0.25;
				d += x_tmp[i][j]*x_tmp[i][j] - x[i][j]*x[i][j];
			}
		}
		tmp = x;
		x = x_tmp;
		x_tmp = tmp;
		k++;
	}
	printf("jacobi - k is: %i\n", k);
	free_A(x_tmp);
	return x;
}

double** parallel2_jacobi( double **A, double** b, int N, int nb_iter, double threshold)
{
	printf("INIT Jacobi (parallel)\n");
	double ** x = dmalloc_2d(N);
	double ** x_tmp = dmalloc_2d(N);
	// used for update
	double ** tmp = 0;
	set_x0(x, N);
	set_x0(x_tmp, N);
	int i, j ,k;
	k = 0;
	double d = INFINITY;
	double d_tmp = INFINITY;
	double delta = 2.0 / ((double)(N-3));
	// parallel for outside - orphaning - d_tmp shared
	#pragma omp parallel private(i,j)
	while ( d_tmp > threshold && k < nb_iter )
	{
		#pragma omp for reduction(+:d)
		for (i = 1; i < N-1; i++)
		{
			for (j = 1; j < N-1; j++)
			{
				x_tmp[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1] + delta * delta * b[i][j])*0.25;
				d += x_tmp[i][j]*x_tmp[i][j] - x[i][j]*x[i][j];
			}
		}
		// code protected with single, d for setting inside for d_tmp for checking
		#pragma omp single
		{
			d_tmp = d;
			d = 0.0;
			tmp = x;
			x = x_tmp;
			x_tmp = tmp;
			k++;
		}
	}
	printf("jacobi - k is: %i\n", k);
	free_A(x_tmp);
	return x;
}