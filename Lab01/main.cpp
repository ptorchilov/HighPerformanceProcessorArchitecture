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

	for (int i = 0; i < M * K * M * K; i++) {
		matrix_C[i] = 0;
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

	printf("%lf", ((double) (end - start)) / CLK_TCK);

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