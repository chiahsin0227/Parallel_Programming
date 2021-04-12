#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <omp.h>
#include <iostream>

#define INF 1073741823

static int Dist[6000][6000];
int n, m;

void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);


int ceil(int a, int b) { return (a + b - 1) / b; }

void cal( int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {

    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    int Round_B = Round * B;
    int max_val = ( ( (Round + 1) * B ) > n )? n : (Round + 1) * B;
    
    switch(block_height){
        case 1:
        {
            int block_internal_start_x = block_start_x * B;
            int block_internal_end_x = (block_start_x + 1) * B;

            if (block_internal_end_x > n) block_internal_end_x = n;

            #pragma omp parallel for schedule(dynamic)
            for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
                
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int k = Round_B; k < max_val; ++k) {
                    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                        int dist_i_k = Dist[i][k];
                        for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                            int val = dist_i_k + Dist[k][j];
                            int disk_i_j = Dist[i][j];
                            Dist[i][j] = val * (val < disk_i_j) + disk_i_j * (val >= disk_i_j);
                        }
                    }
                }
            }
            break;
        }
        default:
        {
            #pragma omp parallel
            {   
                #pragma omp for schedule(dynamic)
                for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {

                    int block_internal_start_x = b_i * B;
                    int block_internal_end_x = (b_i + 1) * B;

                    if (block_internal_end_x > n) block_internal_end_x = n;

                    for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
                        
                        int block_internal_start_y = b_j * B;
                        int block_internal_end_y = (b_j + 1) * B;

                        if (block_internal_end_y > n) block_internal_end_y = n;

                        for (int k = Round_B; k < max_val; ++k) {
                            for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                                int dist_i_k = Dist[i][k];
                                for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                                    int val = dist_i_k + Dist[k][j];
                                    int disk_i_j = Dist[i][j];
                                    Dist[i][j] = val * (val < disk_i_j) + disk_i_j * (val >= disk_i_j);
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
    }
}

void block_FW(int B, int round) {  
    
    for (int r = 0; r < round; ++r) {

        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1); // up
        cal(B, r, r, r + 1, round - r - 1, 1); // down
        cal(B, r, 0, r, 1, r); // left
        cal(B, r, r + 1, r, 1, round - r - 1); // right

        /* Phase 3*/
        cal(B, r, 0, 0, r, r); // upper-left
        cal(B, r, 0, r + 1, round - r - 1, r); // lower-left
        cal(B, r, r + 1, 0, r, round - r - 1); // upper-right
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1); // lower-right
    }
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            Dist[i][j] = INF;
        }
        Dist[i][i] = 0;
        for (int j = i + 1; j < n; ++j) {
            Dist[i][j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

int main(int argc, char** argv) {

    input(argv[1]);
    
    int B = 64;
    int round = ceil(n, B);
    block_FW(B, round);
    output(argv[2]);

}
