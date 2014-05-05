#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppm_helpers.h"

int load_ppm(const char* file, uint8_t** data, unsigned int* w, unsigned int* h, unsigned int* channels) 
{
    FILE *fp = 0;
    char header[PGM_HEADER_SIZE];

    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;

    if((fp = fopen(file, "rb")) == 0) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }
    
    if (fgets(header, PGM_HEADER_SIZE, fp) == 0) {
        fprintf(stderr, "reading PGM header returned NULL\n");
        return -1;
    }

    if (strncmp(header, "P5", 2) == 0) {
        *channels = 1;
    } else if (strncmp(header, "P6", 2) == 0) {
        *channels = 3;
    } else {
        fprintf(stderr, "File is not a PPM or PGM image\n");
        *channels = 0;
        return -1;
    }

    while(i < 3) {
        if (fgets(header, PGM_HEADER_SIZE, fp) == 0) {
            fprintf(stderr, "reading PGM header returned NULL\n");
            return false;
        }

        if(header[0] == '#') {
            continue;
        }

        if(i == 0) {
            i += sscanf(header, "%u %u %u", &width, &height, &maxval);
        } else if (i == 1) {
            i += sscanf(header, "%u %u", &height, &maxval);
        } else if (i == 2) {
            i += sscanf(header, "%u", &maxval);
        }
    }

    if(*data != 0) {
        if (*w != width || *h != height) {
            fprintf(stderr, "Invalid image dimensions\n");
        }
    } else {
        *data = (uint8_t*)malloc(sizeof(uint8_t) * width * height * (*channels));
        *w = width;
        *h = height;
    }

    if (fread(*data, sizeof(unsigned char), width * height * (*channels), fp) == 0) {
        fprintf(stderr, "Read data returned error\n");
    }

    fclose(fp);

    return 0;
}


int load_ppm_alpha(const char *file, uint8_t **data, unsigned int *w, unsigned int *h) {
    uint8_t *idata = 0, *idata_orig, *ptr;
    unsigned int channels;
    int size;

    if (load_ppm(file, &idata, w, h, &channels) == 0) {
        size = *w * *h;
        idata_orig = idata;

        *data = (uint8_t*)malloc(sizeof(uint8_t) * size * 4);
        ptr = *data;

        for (int i = 0; i < size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }

        free(idata_orig);
        return 0;
    } else {
        free(idata);
        return -1;
    }
}


int store_ppm(const char* file, uint8_t* data, unsigned int w, unsigned int h, unsigned int channels) 
{
    if (data == 0 || w < 1 || h < 1) {
        fprintf(stderr, "Invalid arguments\n");
        return -1;
    }

    FILE* fp;

    if ((fp = fopen(file, "wb")) == 0) {
        fprintf(stderr, "Opening file failed\n");
        return -1;
    }

    if (channels == 1){
        fprintf(fp, "P5\n");
    } else if (channels == 3) {
        fprintf(fp, "P6\n");
    } else {
        fprintf(stderr, "Invalid number of channels %d\n", channels);
        return -1;
    }

    fprintf(fp, "%d %d\n255\n", w, h);
   
    if (fwrite(data, sizeof(*data), w * h * channels, fp) == 0) {
        fprintf(stderr, "Writing data failed\n");
        return -1;
    }

    fflush(fp);
    fclose(fp);

    return 0;
}

int store_ppm_alpha(const char *file, uint8_t *data, unsigned int w, unsigned int h) {
    int result, size = w * h;

    uint8_t *ndata = (uint8_t*)malloc(sizeof(uint8_t) * size * 3);
    uint8_t *ptr = ndata;

    for (int i = 0; i < size; i++) {
        *ptr++ = *data++;
        *ptr++ = *data++;
        *ptr++ = *data++;
        data++;
    }

    result = store_ppm(file, ndata, w, h, 3);
    free(ndata);
    return result;
}
