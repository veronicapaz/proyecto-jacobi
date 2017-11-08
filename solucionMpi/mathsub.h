#ifndef _MATHSUB_H_
#define _MATHSUB_H_

void dgemm_seq(int n, int m, double alpha, double *A, double *B, double beta, double *C, int p);
void dnrm2_seq(int n, double *x, double *nrm, int p);
void dgemv_seq(int n, int m, double alpha, double *A, double *b, double beta, double *c, int p);
void dcopy_seq(int n, double *x, double *y);
void daxpy_seq(int n, double alpha, double *x, double *y, int p);


#endif // _MATHSUB_H_