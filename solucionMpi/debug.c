#include "jacobi.h"


void printVec(double* a, int n)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		printf("[ %.8f ]\n", a[i]);
	}
}


void printMat(double* A, int n, int m)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		printf("[ ");
		
		for (j = 0; j < m; j++)
		{
			printf("%.8f ", A[i + j*n]);
		}
		
		printf("]\n");
	}
}