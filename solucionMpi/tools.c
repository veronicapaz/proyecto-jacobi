#include "jacobi.h"


int generateMatrices(double *A, double *b, double *x0, int n)
{
	int i, j;
	srand(time(NULL));
	
	if (A == NULL || b == NULL || x0 == NULL)
	{
		return errno;
	}
	
	// Rellenamos A y b con valores aleatorios,
	// y x0 con todo unos
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			// Forzamos que A sea de diagonal dominante
			A[i + j*n] = 1.0 + (double)(rand() % 10);
			if (i == j) A[i + j*n] += (double)n*10.0;
		}
		
		b[i] = 1.0 + (double)(rand() % 10);
		x0[i] = 1.0;
	}
	
	return 0;
}


void saveMatrix(char *name, double *x, int n, int ndim)
{
	char fname[128];
	strcpy(fname, name);
	strcat(fname, ".dlm");

	FILE *f = fopen(fname, "w+");
	int i, j;
	
	if (f == NULL)
	{
		fprintf(stderr, "Error saving result: %s\n", strerror(errno));
		return;
	}
	
	if (ndim == 1)
	{
		for (i = 0; i < n; i++)
		{
			fprintf(f, "%.12f\n", x[i]);
		}
	}
	else if (ndim == 2)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				fprintf(f, "%.12f ", x[i + j*n]);
			}
			
			fprintf(f, "\n");
		}
	}
	
	fclose(f);
}