#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <math.h>
#include <malloc.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <signal.h>
#include <omp.h>
#include <string.h>

#if defined(__INTEL_COMPILER)
#include <mkl.h>
#endif

#if defined(MPI)
#include <mpi.h>
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Este valor sobreescribe el número de hilos OpenMP
// a utilizar en las secciones paralelas.
//
// Comentar para que se utilicen tantos hilos como
// procesadores tenga la máquina.
//#define FORCE_THREADS 2

// Errores devueltos por el algoritmo de Jacobi:
// 
// > ENONDOM: la matriz de términos independientes
// 		      del sistema de ecuaciones no es de
//	          diagonal estrictamente dominante, por
//            lo que no está asegurada la convergencia.
enum
{
	ENONDOM = -1
};

// Iteración de Jacobi
// - Esta función ejecuta la iteración de Jacobi hasta
//   que la solución obtenida tenga una cota de error
//   inferior a la especificada.
//
// - En función del tamaño del problema, utilizará nucleos
//   computacionales paralelos o no.
//
// Parámetros de entrada:
// > (double[][]) A: matriz de términos independientes del sistema
//    		         de ecuaciones.
//
// > (double[]) B: vector de coeficientes del sistema de ecuaciones.
//
// > (double[]) x0: vector con la aproximación inicial a la solución
//     		        del sistema.
//
// > (double) conv: valor de error mínimo para el que debe de converger
// 				    el sistema.
//
// > (int) n: tamaño del lado de la matriz A y longitud de los vectores b
//			  y x0.
//
// Parámetros de salida:
// > (int): devuelve el número de iteraciones que han sido necesarias para
//	        obtener la solución, o un error en caso de fallo.
int jacobi(double *A, double *b, double *x0, double conv, int n);
int jacobi_mpi(double *A, double *b, double *x0, double conv, int n, int rank, int size, char* hname);


double jaciter(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n);

static inline double 
__jaciter_kernel_sequential(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n);

static inline double 
__jaciter_kernel_parallel(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n, int p);


void getrd(double *Dinv, double *R, double *A, double *b, double *C, int n, int m);
static inline void __getrd_kernel_sequential(double *Dinv, double *R, double *A, double *b, double *C, int n, int m);
static inline void __getrd_kernel_parallel(double *Dinv, double *R, double *A, double *b, double *C, int n, int m, int p);

int isdom(double *A, int n);
static inline int __isdom_kernel_sequential(double *A, int n);
static inline int __isdom_kernel_parallel(double *A, int n, int p);


double* diaginv(double *A, int n, double *diag);
static inline void __diaginv_kernel_sequential(double *A, int n, double *diag);
static inline void __diaginv_kernel_parallel(double *A, int n, double *diag, int p);


void printMat(double* A, int n, int m);
void printVec(double* a, int n);

int generateMatrices(double *A, double *b, double *x0, int n);
void saveMatrix(char *name, double *x, int n, int ndim);


#endif // _JACOBI_H_