
#include "../jacobi.h"
#include "../mathsub.h"

int jacobi_mpi(double *A, double *b, double *x0, double conv, int n, int rank, int size, char* hname)
{
	double *Dinv = NULL, *C = NULL, *xkp1 = NULL,
	*xconv = NULL, *R = NULL, e = conv + 1.0;
	int matdm = 0, k = 0, chunk = 0, run = 1, i = 0, p = 1;
	MPI_Status status;
	
	// Calculamos el tamaño de bloque. Dividimos el tamaño del 
	// problema (n) entre p - 1 nodos, ya que el nodo cero no
	// realiza procesamiento por bloques.
	chunk = n/(size-1);
	
	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	//
	// Sólo lo comprobamos en el nodo 0, en caso de no cumplir
	// la condición se cancela la ejecución
	if (rank == 0)
	{
		if (!isdom(A, n))
		{
			fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": 'A' matrix is non dominant.\n", hname);
			MPI_Abort(MPI_COMM_WORLD, ENONDOM);
		}
	}
	
	// Desde el nodo cero se envía la matriz A, los vectores
	// b y x0.
	MPI_Bcast(A, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x0, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// Las matrices D^-1 y L+U se obtienen en el nodo cero
	// y se envían a el resto de nodos.
	Dinv = (double*) malloc(sizeof(double)*n);
	R = (double*) malloc(sizeof(double)*n*n);
	C = (double*) malloc(sizeof(double)*n);
	xkp1 = (double*) malloc(sizeof(double)*n);
	xconv = (double*) malloc(sizeof(double)*n);
	
	// Ha ocurrido un error al reservar memoria en uno de los nodos:
	// Cancelamos la ejecucción del programa.
	if (Dinv == NULL || R == NULL || C == NULL || xkp1 == NULL || xconv == NULL)
	{
		fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": %s\n", hname, strerror(errno));
		MPI_Abort(MPI_COMM_WORLD, errno);
	}
	
	// Obtenemos la inversa de la diagonal de A,
	// la matriz L+U y el vector C en el nodo cero. 
	// Una vez obtenidas, las mandamos al resto de
    // nodos.
	if (rank == 0)
	{
		getrd(Dinv, R, A, b, C, n, n);
		
		MPI_Bcast(Dinv, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(R, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		for (i = 1; i < size; i++)
		{
			MPI_Send(C, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
		
		p = omp_get_max_threads();
		
		#ifdef FORCE_SEQUENTIAL
		p = 1;
		#elif FORCE_OPENMP
			p = omp_get_max_threads();
			
			#ifdef FORCE_THREADS
			p = FORCE_THREADS;
			#endif
		#endif
	}
	else
	{
		MPI_Bcast(Dinv, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(R, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Recv(C, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
	}
	
	dcopy_seq(n, x0, xkp1);
	
	// Iteramos hasta alcanzar la razón de convergencia.
	// Cada nodo obtendrá la parte que le corresponda de
	// la solución.
	//
	// Para obtener la razón de convergencia:
	// e = sqrt(sum(Ax-b)^2)
	//
	// Se calculará en cada nodo la parte correspondiente
	// de la operación de álgebra matricial, el cuadrado de
	// cada elemento y su suma.
	//
	// Posteriormente se enviará cada resultado al nodo cero,
	// el cual sumará todos los valores y obtendrá la raíz,
	// calculando así la norma del vector.
	//
	// Por tanto, el nodo cero será el encargado de decidir si se
	// debe continuar con la iteración o si se ha alcanzado la razón
	// de convergencia deseada. El resto de nodos esperarán su respuesta.
	while (run)
	{
		if (rank == 0)
		{
			// Recibimos el parámetro de convergencia de cada nodo y calculamos
			// la raíz cuadrada para obtener la 2-norma
			for (i = 1; i < size; i++)
			{
				MPI_Bcast(x0 + chunk*(i-1), chunk, MPI_DOUBLE, i, MPI_COMM_WORLD);
			}
			
			dgemv_seq(n, n, 1.0, A, x0, 0.0, xconv, p);
			daxpy_seq(n, -1.0, b, xconv, p);
			dnrm2_seq(n, xconv, &e, p);
			
			#ifdef DEBUG
			printf("(k = %d) e = %e / conv = %e\n", k, e, conv);
			printVec(x0, n);
			#endif
			
			// Si aún no hemos alcanzado el valor de convergencia requerido,
			// continuamos.
			//
			// Avisamos al resto de nodos de si deben continuar o si deben
			// de acabar.
			if (e < conv)
			{
				run = 0;
			}
			else
			{
				k++;
			}
			
			MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else
		{
			// Calculamos la solución local y mandamos el parámetro de convergencia
			jaciter(A, b, R, C, Dinv, x0, xkp1, xconv, n);
			
			// Mandamos los bloques de x(k) al resto de nodos
			for (i = 1; i < size; i++)
			{
				MPI_Bcast(x0 + chunk*(i-1), chunk, MPI_DOUBLE, i, MPI_COMM_WORLD);
			}

			// Obtenemos del nodo cero la bandera que indica si debemos continuar
			// calculando o si por contra ya se ha llegado al valor de convergencia
			// deseado
			MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Liberamos la memoria utilizada
	free(C);
	free(xkp1);
	free(xconv);
	free(R);
	free(Dinv);
	
	return k;
}