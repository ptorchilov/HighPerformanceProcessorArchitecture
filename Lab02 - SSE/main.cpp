#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <conio.h>
#include <emmintrin.h>

const int M = 175, K = 6;

double* generate_matrix();
double rand_double(double, double);

int main()
{
	double *matrix_A = new double[M * K * M * K];
	double *matrix_B = new double[M * K * M * K];
	double *matrix_C = new double[M * K * M * K];
    double *matrix_D = new double[M * K * M * K];

	for (int i = 0; i < M * K * M * K; i++) {
		matrix_C[i] = 0;
        matrix_D[i] = 0;
	}

	srand(time(NULL));

	matrix_A = generate_matrix();
	matrix_B = generate_matrix();
		
	clock_t end, start = clock();
	
	for (int i = 0; i < M * K; i++) {
      for(int j = 0; j < M * K; j++) {
		for(int n = 0; n < M * K; n++) {
				matrix_C[M * K * i + n] += matrix_A[M * K * i + j] * matrix_B[M * K * j + n];
			}
		}
	}

    end = clock();

	printf("Vectorization = %lf\n", ((double) (end - start)) / CLK_TCK);

    printf("Control values: %lf %lf %lf\n", matrix_C[10], matrix_C[131], matrix_C[516]);

	__m128d r1, r2, r3, r4, r5, r6, r7, r8;

    start = clock();

	for (int i = 0; i < M * K; i++) {
      for(int j = 0; j < M * K; j++) {
		for(int n = 0; n < M * K; n += 2) {
				r1 = _mm_load1_pd(&matrix_A[M * K * i + j]);
				r2 = _mm_load_pd(&matrix_B[M * K * j + n]); 
               
				r3 = _mm_mul_pd(r1, r2);

				r4 = _mm_load_pd(&matrix_D[M * K * i + n]);
                _mm_store_pd(&matrix_D[M * K * i + n], _mm_add_pd(r4, r3));
            }
		}
	}

    end = clock();

    printf("SSE2 Intrinsics = %lf\n", ((double) (end - start)) / CLK_TCK);

    printf("Control values: %lf %lf %lf\n", matrix_D[10], matrix_D[131], matrix_D[516]);

    getch();

    return 0;
}

double* generate_matrix()
{
	double* matrix = new double[M * K * M * K];

	for(int i = 0; i < M * K; i++) {
		for(int j = 0; j < M * K; j++) {
            matrix[M * K * i + j] = rand_double(0, 1);
        }
    }

	return matrix;
}

double rand_double(double min, double max)
{
    double value = (double) rand() / RAND_MAX;

    return min + value * (max - min);
}