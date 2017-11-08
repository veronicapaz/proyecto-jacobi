#include "jacobi.h"
#include <sys/time.h>
#include <locale.h>
#include <mpi.h>
#include "debug.c"
#include "iter.c"
#include "mathsub.c"
#include "mathsub.h"
#include "isdom.c"
#include "rd.c"
#include "tools.c"

double conv = 0.01;
	int n, i, j, k, save = 0, rank, size, hlen;
	struct timeval t0, tf;
	double ep;
	char hname[MPI_MAX_PROCESSOR_NAME];

int main (int argc, char* argv[])
{
	MPI_Status info;
// Inicializamos MPI y obtenemos el número de nodo
	// local, la cantidad de nodos en ejecucción y el
	// hostname del nodo local.
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(hname, &hlen);

	double *A = NULL, *b = NULL, *x0 = NULL;
	
	if (argc < 2)
	{
		printf("Usage: jacobi [matrix_size] [--save]\n");
		
		MPI_Finalize();
		exit(0);
	}
	else if (argc == 3)
	{
		if (strcmp("--save", argv[2]) == 0)
		{
			save = 1;
		}
		else
		{
			printf("Unrecognized option \"%s\".\n", argv[2]);
			printf("Usage: jacobi [matrix_size] [--save]\n");
			
			MPI_Finalize();
			exit(0);
		}
	}
	
	// Obtenemos el tamaño de las matrices del problema a
	// partir de los parámetros de entrada.
	n = atoi(argv[1]);
	
	A = (double*) malloc(sizeof(double)*n*n);
	b = (double*) malloc(sizeof(double)*n);
	x0 = (double*) malloc(sizeof(double)*n);
	
	// Creamos las matrices del problema de tamaño 1*n o
	// n*n
	generateMatrices(A,b,x0,n);
	
	// Para el nodo cero, los vectores y matrices del
	// sistema contendrán los valores que tendrá que repartir
	// al resto de nodos.
	//
	// Para el resto de nodos, las matrices estarán vacías, y tendrán
	// que rellenarlas a partir de los datos que envíe el nodo cero.
	gettimeofday(&t0, NULL);
	k = jacobi_mpi(A, b, x0, conv, n, rank, size, hname);
	gettimeofday(&tf, NULL);
	
	if (k < 0)
	{
		fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": %s\n", hname, strerror(k));
	}
	
	ep = (tf.tv_sec - t0.tv_sec) + (tf.tv_usec - t0.tv_usec) / 1000000.0;
	
	// Borramos los datos inicializados al inicio del problema
	// y mostramos las estadísticas (tamaño matriz, iteraciones y tiempo).
	// Si se ha especificado, guardamos el resultado como archivo DLM.
	if (rank == 0)
	{
		setlocale(LC_NUMERIC, "es_ES.UTF-8");
		printf("%d;%d;%.6f\n", n, k, ep);
		
		if (save)
		{
			saveMatrix("x", x0, n, 1);
			saveMatrix("A", A, n, 2);
			saveMatrix("b", b, n, 1);
		}
	}
	
	free(A);
	free(b);
	free(x0);
	
	// Finalizamos MPI y salimos
	MPI_Finalize();
	return 0;
}