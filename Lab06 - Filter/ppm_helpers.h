#ifndef PPM_HELPERS_H
#define PPM_HELPERS_H

#include <cuda.h>
#include <stdint.h>

#define PGM_HEADER_SIZE 0x40

int load_ppm(const char* file, uint8_t** data, unsigned int* w, unsigned int* h, unsigned int* channels);
int load_ppm_alpha(const char *file, uint8_t **data, unsigned int *w,unsigned int *h);

int store_ppm(const char* file, uint8_t* data, unsigned int w, unsigned int h, unsigned int channels);
int store_ppm_alpha(const char *file, uint8_t *data, unsigned int w, unsigned int h);

#endif //PPM_HELPERS_H

