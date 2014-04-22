#include <stdio.h>
#include <conio.h>
#include <intrin.h>
#pragma intrinsic(__rdtsc)

int* block;
int BLOCK_SIZE =  32 * 2048; 
int OFFSET = 1 * 1024 * 1024;
int const MAX_WAY = 20; 

void fill_block(int fragment, int offset, int n)
{
	for(int i = 0; i < fragment; i++) {
		for(int j = 0; j < n; j++) {
			if(j < n - 1) {
				block[j * offset + i] = (j + 1) * offset + i;
			} else if(i < fragment - 1) {
				block[j * offset + i] = i + 1;
			} else {
				block[j * offset + i] = 0;
			}
		}			
	}
}

void print_chart(double* timer_array) 
{
	double value = 0.0;

	for (int i = 0; i < MAX_WAY; i++) {
		printf("\n%02i ", i + 1);
		
		do {
			value += 50;
			printf("*");

		} while (value <= timer_array[i]);

		value = 0.0;
		printf(" - time to read - %.1lf", timer_array[i]);
	}
}


int main()
{
	block = (int*) malloc (OFFSET * MAX_WAY * sizeof(int));

	double timer_array[MAX_WAY];	

	for(int i = 1; i <= MAX_WAY ; i++) {
		register unsigned long long begin = 0.0, end = 0.0;
		register int index = 0;

		fill_block(BLOCK_SIZE / i, OFFSET, i);		
				
		begin = __rdtsc();

		for(int i = 1; i <= MAX_WAY; i++) {			
			do {
				index = block[index];
			} while (index != 0);
		}
		
		end = __rdtsc();
		
		timer_array[i - 1] = (end - begin) / 32768;		
	}

	print_chart(timer_array);

	_getch();

	return 0;
}
