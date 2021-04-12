#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct{
    int num_thread;
    int thread_id;
    int *image;
    double left;
    double right;
    double upper;
    double lower;
    int height;
    int width;
    int iters;
}Arg;

void* threadFunc(void* argument)
{
    
    Arg* arg = (Arg*) argument;
    int num_thread = arg->num_thread;
    int thread_id = arg->thread_id;
    int *image = arg->image;
    double left = arg->left;
    double right = arg->right;
    double upper = arg->upper;
    double lower = arg->lower;
    int height = arg->height;
    int width = arg->width;
    int iters = arg->iters;
    
    double tmp = ((right - left) / width);
    double x0[width];
    double y0[height];
    int j_x_width[height];
    #pragma GCC ivdep
    for (int i = 0; i < width; ++i) 
        x0[i] = i * tmp + left;
    
    tmp = ((upper - lower) / height);

    for (int j = 0; j < height; ++j) {
        j_x_width[j] = j * width;
        y0[j] = j * tmp + lower;
    }

    int total = height*width;
    int count = thread_id + num_thread;
    int index[2][2] = { {thread_id/width,  thread_id % width}, {count/width, count % width} };
    double y0_[2] = {y0[index[0][0]], y0[index[1][0]]};
    double x0_[2] = {x0[index[0][1]], x0[index[1][1]]};
    int repeats[2] = {0, 0};
    double x[2] = {0, 0};
    double y[2] = {0, 0};
    double xx[2] = {0, 0};
    double yy[2] = {0, 0};
    int now[2] = {thread_id, count};
    double length_squared[2] = {0, 0};
    while (count < total) {
        while (length_squared[0] < 4 && length_squared[1] < 4 && repeats[0] < iters && repeats[1] < iters) {
            for (int k = 0; k < 2; ++k){
                y[k] = 2 * x[k] * y[k] + y0_[k];
                x[k] = xx[k] - yy[k] + x0_[k];
                xx[k] = x[k] * x[k];
                yy[k] = y[k] * y[k];
                length_squared[k] = xx[k] + yy[k];
                ++(repeats[k]);
            }
        }
        if(length_squared[0] >= 4 || repeats[0] >= iters){
            image[ j_x_width[ index[0][0] ] + index[0][1] ] = repeats[0];
            count += num_thread;
            repeats[0] = 0;
            x[0] = y[0] = xx[0] = yy[0] = length_squared[0] = 0;
            index[0][0] = count/width;
            index[0][1] = count%width;
            y0_[0] = y0[ index[0][0] ];
            x0_[0] = x0[ index[0][1] ];
            now[0] = count;
        } 
        if(length_squared[1] >= 4 || repeats[1] >= iters) {
            image[ j_x_width[ index[1][0] ] + index[1][1] ] = repeats[1];
            count += num_thread;
            repeats[1] = 0;
            x[1] = y[1] = xx[1] = yy[1] = length_squared[1] = 0;
            index[1][0] = count/width;
            index[1][1] = count%width;
            y0_[1] = y0[ index[1][0] ];
            x0_[1] = x0[ index[1][1] ];
            now[1] = count;
        }
    }
    if(now[0] < total){
        while (length_squared[0] < 4 && repeats[0] < iters){
            y[0] = 2 * x[0] * y[0] + y0_[0];
            x[0] = xx[0] - yy[0] + x0_[0];
            xx[0] = x[0] * x[0];
            yy[0] = y[0] * y[0];
            length_squared[0] = xx[0] + yy[0];
            ++repeats[0];
        }
        image[ j_x_width[ index[0][0] ] + index[0][1] ] = repeats[0];
    } 
    if(now[1] < total) {
        while (length_squared[1] < 4 && repeats[1] < iters){
            y[1] = 2 * x[1] * y[1] + y0_[1];
            x[1] = xx[1] - yy[1] + x0_[1];
            xx[1] = x[1] * x[1];
            yy[1] = y[1] * y[1];
            length_squared[1] = xx[1] + yy[1];
            ++repeats[1];
        }
        image[ j_x_width[ index[1][0] ] + index[1][1] ] = repeats[1];
    }
    return NULL;
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = height; y--; ) {
        memset(row, 0, row_size);
        int tmp = y * width;
        for (int x = 0; x < width; ++x) {
            //printf("[%d, %d] = %d\n", y, x, buffer[y*width+x]);
            int p = buffer[tmp + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p % 16) << 4;
                } else {
                    color[0] = (p % 16) << 4;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    unsigned int num_threads = CPU_COUNT(&cpu_set);

    pthread_t threads[num_threads];
	unsigned int t;
	for (t = 0; t < num_threads; ++t) {
		Arg *arg = new Arg();
        arg->num_thread = num_threads;
        arg->thread_id = t;
        arg->image = image;
        arg->left = left;
        arg->right = right;
        arg->upper = upper;
        arg->lower = lower;
        arg->height = height;
        arg->width = width;
        arg->iters = iters;
        pthread_create(&threads[t], NULL, threadFunc, (void*)arg);
    }

    for (t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
