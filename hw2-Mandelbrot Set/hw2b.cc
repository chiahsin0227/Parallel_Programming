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
#include <mpi.h>
#include <omp.h>

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
    int width3 = 3 * width;
    size_t row_size = width3 * sizeof(png_byte);
    png_bytep row = (png_bytep) malloc (row_size);
    for (int y = height; y--; ) {
        memset(row, 0, row_size);
        int width_x_y = y * width;
        for (int x = 0; x < width3; x+=3) {
            int p = buffer[width_x_y++];
            png_bytep color = row + x;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 << 4;
                } else {
                    color[0] = p % 16 << 4;
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
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int num_threads = CPU_COUNT(&cpu_set);

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

    MPI_Init(&argc,&argv);
	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int revcount[size];
    int displs[size] = {0};

    int h = height / size;
    int remain = height % size;
    revcount[0] = (remain == 0)? h * width : ( h + 1 ) * width;
    
    for(int i = 1; i < size; ++i){
        revcount[i] = (i < remain)? ( h + 1 ) * width : h * width;
        displs[i] = displs[i-1] + revcount[i-1];
    }


    /* mandelbrot set */
    double tmp_x = (right - left) / width;
    double x_[width+2];
    #pragma vector aligned
    for (int i = 0; i < width; ++i) 
        x_[i] = i * tmp_x + left;

    double tmp_y = (upper - lower) / height;
    
    int part_height = (rank < remain)? h + 1 : h;
    int *part_image = (int*) malloc (width * part_height * sizeof(int));
       
    #pragma omp parallel num_threads(num_threads) 
    {
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < part_height; ++j) {
            double y0 = (rank + j * size) * tmp_y + lower;
            int *prev_pixel = part_image + j * width;
            int cnt = 1;
            int repeats[2] = {0, 0};
            double x[2] = {0, 0};
            double y[2] = {0, 0};
            double xx[2] = {0, 0};
            double yy[2] = {0, 0};
            double length_squared[2] = {0, 0};
            int now[2] = {0, 1};
            double x0[2] = {x_[0], x_[1]};
            while(cnt < width) {
                while (length_squared[0] < 4 && length_squared[1] < 4 && repeats[0] < iters && repeats[1] < iters) {
                    #pragma vector aligned
                    for (int k = 0; k < 2; ++k){
                        y[k] = 2 * x[k] * y[k] + y0;
                        x[k] = xx[k] - yy[k] + x0[k];
                        xx[k] = x[k] * x[k];
                        yy[k] = y[k] * y[k];
                        length_squared[k] = xx[k] + yy[k];
                        ++(repeats[k]);
                    }
                }
                if(repeats[0] >= iters || length_squared[0] >= 4){
                    ++cnt;
                    *(prev_pixel + now[0]) = repeats[0];
                    repeats[0] = 0;
                    x[0] = y[0] = xx[0] = yy[0] = length_squared[0] = 0;
                    now[0] = cnt;
                    x0[0] = x_[cnt];
                }
                if(repeats[1] >= iters || length_squared[1] >= 4){
                    ++cnt;
                    *(prev_pixel + now[1]) = repeats[1];
                    repeats[1] = 0;
                    x[1] = y[1] = xx[1] = yy[1] = length_squared[1] = 0;
                    now[1] = cnt;
                    x0[1] = x_[cnt];
                }
            }
            if(now[0] < width){
                while(repeats[0] < iters && length_squared[0] < 4){
                    y[0] = 2 * x[0] * y[0] + y0;
                    x[0] = xx[0] - yy[0] + x0[0];
                    xx[0] = x[0] * x[0];
                    yy[0] = y[0] * y[0];
                    length_squared[0] = xx[0] + yy[0];
                    ++(repeats[0]);
                }
                *(prev_pixel + now[0]) = repeats[0];
            }
            if(now[1] < width){
                while(repeats[1] < iters && length_squared[1] < 4){
                    y[1] = 2 * x[1] * y[1] + y0;
                    x[1] = xx[1] - yy[1] + x0[1];
                    xx[1] = x[1] * x[1];
                    yy[1] = y[1] * y[1];
                    length_squared[1] = xx[1] + yy[1];
                    ++(repeats[1]);
                }
                *(prev_pixel + now[1]) = repeats[1];
            }

        }
    }
    


    /* allocate memory for image */
    int* gather_image = (int*)malloc(width * height * sizeof(int));
    assert(gather_image);

    /* gather all the part image results */
    MPI_Gatherv(part_image, revcount[rank], MPI_INT, gather_image, revcount, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /* change the position to the right position */
    if(rank == 0){
        
        int* image = (int*)malloc(width * height * sizeof(int));
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for schedule(dynamic)
            for(int i = 0; i < size ; ++i){
                int position = displs[i];
                for(int j = i, _j = 0; j < height; j += size, ++_j){
                    int prev_pixel_0 = width * j;
                    int prev_pixel_1 = position + _j * width;
                    #pragma vector aligned
                    for(int k = 0; k < width; ++k){
                        image[prev_pixel_0 + k] = gather_image[prev_pixel_1 + k];
                    }
                }
            }
        }
        
        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
        free(image);
    }

    
    
    MPI_Finalize();
}
