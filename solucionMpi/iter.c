#include "jacobi.h"
#include "mathsub.h"


double jaciter(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n)
{
	int p, chunk, minsize;
	double e;
	
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
		e = __jaciter_kernel_sequential(A, b, R, C, Dinv, xk, xkp1, xconv, n);
	}
	else
	{
		// Paralelo
		e = __jaciter_kernel_parallel(A, b, R, C, Dinv, xk, xkp1, xconv, n, p);
	}
	
	return e;
}

static inline double 
__jaciter_kernel_sequential(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n)
{
	int rank, size, stride, m, i;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	m = n;
	n = m/(size-1);
	stride = n*(rank-1);
	
	dgemv_seq(m, m, 1.0, R, xk, 0.0, xkp1, 1);
		
	for (i = 0; i < n; i++)
	{
		xkp1[i + stride] *= -Dinv[i + stride];
		xkp1[i + stride] += C[i + stride];
	}
	
	dcopy_seq(n, xkp1 + stride, xk + stride);
	return 0.0;
}


static inline double 
__jaciter_kernel_parallel(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n, int p)
{
	int rank, size, stride, m, i;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	#ifdef FORCE_THREADS
	p = FORCE_THREADS;
	#endif
	omp_set_num_threads(p);
	
	m = n;
	n = m/(size-1);
	stride = n*(rank-1);
	
	dgemv_seq(m, m, 1.0, R, xk, 0.0, xkp1, p);
	
	#pragma omp parallel for schedule(static) private(i) shared(xkp1, Dinv, C, n)
	for (i = 0; i < n; i++)
	{
		xkp1[i + stride] *= -Dinv[i + stride];
		xkp1[i + stride] += C[i + stride];
	}
	
	dcopy_seq(n, xkp1 + stride, xk + stride);
	return 0.0;
}