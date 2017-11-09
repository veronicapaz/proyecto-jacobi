#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gauss.h"
#include "utils.h"

double** gauss( double **A, double** b, int N, int nb_iter, double threshold)
{
	printf("INIT Gauss-Seidel (sequential)\n");
	double ** x_tmp = dmalloc_2d(N);
	set_x0(x_tmp, N);
	int i, j, k;
	k = 0;
       	double d = INFINITY;
	double delta = 2.0 / ((double)(N-3));
	double last_var = 0.0;
        while ( d > threshold && k < nb_iter )
	{
		d = 0.0;
		for (i = 1; i < N-1; i++)
		{
			for (j = 1; j < N-1; j++)
			{
				last_var = x_tmp[i][j];
				x_tmp[i][j] = (x_tmp[i-1][j] + x_tmp[i+1][j] + x_tmp[i][j-1] + x_tmp[i][j+1] + delta * delta * b[i][j])/4.0;
				d += x_tmp[i][j]*x_tmp[i][j] - last_var*last_var;
			}
		}
                k++;
	}
        printf("gauss-seidel - k is: %i\n", k);
        return x_tmp;
}