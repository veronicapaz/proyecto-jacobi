#include "mathsub.h"
#include <math.h>
#include <string.h>

void dnrm2_seq(int n, double *x, double* nrm, int p)
{
	*nrm = 0.0;
	int i, nc, pr;
	double *tempnrm = (double*) malloc(sizeof(double)*p);
	
	if (n >= 4)
	{
		nc = n - (n % 4);
		
		if (p == 1)
		{
			for (i = 0; i < nc; i += 4)
			{
				*nrm += x[i]*x[i];
				*nrm += x[i+1]*x[i+1];
				*nrm += x[i+2]*x[i+2];
				*nrm += x[i+3]*x[i+3];
			}
		}
		else
		{
			#pragma omp parallel private(i) shared(x, nc, tempnrm, pr)
			{
				pr = omp_get_thread_num();
				tempnrm[pr] = 0.0;
			
				#pragma omp for schedule(static)
				for (i = 0; i < nc; i += 4)
				{
					tempnrm[pr] += x[i]*x[i];
					tempnrm[pr] += x[i+1]*x[i+1];
					tempnrm[pr] += x[i+2]*x[i+2];
					tempnrm[pr] += x[i+3]*x[i+3];
				}
			}
			
			for (i = 0; i < p; i++)
			{
				*nrm += tempnrm[i];
			}
		}
		
		if (nc != n)
		{
			for (i = nc; i < n; i++)
			{
				*nrm += x[i]*x[i];
			}
		}
	}
	else
	{
		for (i = 0; i < n; i++) *nrm += x[i]*x[i];
	}
	
	free(tempnrm);
	*nrm = sqrt(*nrm);
}


void dgemm_seq(int n, int m, double alpha, double *A, double *B, double beta, double *C, int p)
{
	int i, k, j;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			C[i + j*n] = 0.0;
			
			for (k = 0; k < m; k++) 
			{
				C[i + j*n] += alpha * A[i + k*m] * B[k + j*m];
			}
		}
    }
}


void dgemv_seq(int n, int m, double alpha, double *A, double *b, double beta, double *c, int p)
{
	int i, j, mc;
	
	if (p == 1)
	{
		for (i = 0; i < n; i++)
		{
			c[i] = 0.0;
			
			mc = m - (m % 4);
			
			if (m >= 4)
			{
				for (j = 0; j < mc; j += 4)
				{
					c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
					c[i] += alpha * A[i + (j+1)*n] * b[j+1] + beta * c[i];
					c[i] += alpha * A[i + (j+2)*n] * b[j+2] + beta * c[i];
					c[i] += alpha * A[i + (j+3)*n] * b[j+3] + beta * c[i];
				}
				
				if (mc != m)
				{
					for (i = mc; i < m; i++)
					{
						c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
					}
				}
			}
			else
			{
				for (j = 0; j < m; j++)
				{
					c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
				}
			}
		}
	}
	else
	{
		#pragma omp parallel for schedule(static) private(i, j) shared(c, A, b, n, m, mc, alpha, beta)
		for (i = 0; i < n; i++)
		{
			c[i] = 0.0;
			
			mc = m - (m % 4);
			
			if (m >= 4)
			{
				for (j = 0; j < mc; j += 4)
				{
					c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
					c[i] += alpha * A[i + (j+1)*n] * b[j+1] + beta * c[i];
					c[i] += alpha * A[i + (j+2)*n] * b[j+2] + beta * c[i];
					c[i] += alpha * A[i + (j+3)*n] * b[j+3] + beta * c[i];
				}
				
				if (mc != m)
				{
					for (i = mc; i < m; i++)
					{
						c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
					}
				}
			}
			else
			{
				for (j = 0; j < m; j++)
				{
					c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
				}
			}
		}
	}
}


void dcopy_seq(int n, double *x, double *y)
{
	memcpy(y, x, sizeof(double) * n);
}

void daxpy_seq(int n, double alpha, double *x, double *y, int p)
{
	int i, nc;
	
	if (n >= 4)
	{
		nc = n - (n % 4);
		
		if (p == 1)
		{
			for (i = 0; i < nc; i += 4)
			{
				y[i] += alpha * x[i];
				y[i+1] += alpha * x[i+1];
				y[i+2] += alpha * x[i+2];
				y[i+3] += alpha * x[i+3];
			}
		}
		else
		{
			#pragma omp parallel for schedule(static) private(i) shared(x, y, nc, alpha)
			for (i = 0; i < nc; i += 4)
			{
				y[i] += alpha * x[i];
				y[i+1] += alpha * x[i+1];
				y[i+2] += alpha * x[i+2];
				y[i+3] += alpha * x[i+3];
			}
		}
		
		if (nc != n) {
			for (i = nc; i < n; i++) y[i] += alpha * x[i];
		}
	}
	else
	{
		for (i = 0; i < n; i++) y[i] += alpha * x[i];
	}
}