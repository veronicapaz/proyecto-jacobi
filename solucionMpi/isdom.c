#include "jacobi.h"


int isdom(double *A, int n)
{
	int p, chunk, minsize, dom;
	
	p = omp_get_max_threads();
	minsize = BLOCK_SIZE;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	chunk = minsize - 1;
	#elif FORCE_OPENMP
	chunk = minsize + 1;
	#endif
	
	if (chunk < minsize)
	{
		// Secuencial
		dom = __isdom_kernel_sequential(A, n);
	}
	else
	{
		// Paralelo
		dom = __isdom_kernel_parallel(A, n, p);
	}
	
	return dom;
}


static inline int __isdom_kernel_sequential(double *A, int n)
{
	int dom = 1, i, j;
	double diag = 0.0, rowmn = 0.0, rowmn2 = 0.0;
	
	for (i = 0; i < n; i++)
	{
		diag = A[i + i*n];
	
		for (j = 0; j < n; j++)
		{
			if (j != i)
			{
				rowmn = A[i + j*n];
				rowmn2 += rowmn * rowmn;
			}
		}
		
		if (diag < sqrt(rowmn2))
		{
			dom = 0;
		}
	}
	
	return dom;
}


static inline int __isdom_kernel_parallel(double *A, int n, int p)
{
	int dom = 1, i, j;
	double diag = 0.0, rowmn = 0.0, rowmn2 = 0.0;
	#ifdef FORCE_THREADS
	p = FORCE_THREADS;
	#endif
	omp_set_num_threads(p);
	
	#pragma omp parallel for schedule(static) private(i, j, diag, rowmn, rowmn2) shared(n, A, dom)
	for (i = 0; i < n; i++)
	{
		diag = A[i + i*n];
		
		for (j = 0; j < n; j++)
		{
			if (j != i)
			{
				rowmn = A[i + j*n];
				rowmn2 += rowmn * rowmn;
			}
		}
		
		if (diag < sqrt(rowmn2))
		{
			dom = 0;
		}
	}
	
	return dom;
}
