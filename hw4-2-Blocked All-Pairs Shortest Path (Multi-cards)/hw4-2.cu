#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <omp.h>

const int INF = ((1 << 30) - 1); 
inline void input(char *inFileName);
inline void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void phase1(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* d_dist, int pitch_int);
__global__ void phase2(int B, int Round, int n, int* d_dist, int pitch_int);
__global__ void phase3(int B, int Round, int n, int* d_dist, int pitch_int, int thread_num, int total_round);

int n, m, v;   // Number of vertices, edges
int* Dist;
int* d_dist[2];
size_t pitch;

int main(int argc, char* argv[])
{   
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

inline void input(char* infile) { 
    FILE* file = fopen(infile, "rb"); 
    fread(&v, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    if(v%64) n = v + (64 - v%64);
    else n = v;
    cudaMallocHost( &Dist, sizeof(int)*(n*n)); 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) { 
        int in = i*n;
        #pragma simd
        for (int j = 0; j < i; ++j) {
                Dist[in + j] = INF;
        }
        Dist[in + i] = 0;
        for (int j = i + 1; j < n; ++j) {
                Dist[in + j] = INF;
        }
    } 
    m += m << 1;

    int *pair = (int*)malloc(sizeof(int)*m);
    fread(pair, sizeof(int), m, file);
    #pragma simd
    for (int i = 0; i < m; i += 3) {
        Dist[pair[i]*n+pair[i+1]] = pair[i+2];
    }
    fclose(file);
    free(pair);
}

void output(char *outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    int *d = Dist;
    for (int i = 0; i < v; ++i) {
        fwrite(d, sizeof(int), v, outfile);
        d +=n;
    }
    fclose(outfile);
}

int ceil(int a, int b) {
    return (a + b -1)/b;
}

void block_FW(int B)
{
    int round = ceil(n, B);
    dim3 grid2(round-1, 2);
    dim3 grid3(round-1, (round/2)+1);
    dim3 block(64, 16);
    dim3 block2(32, 32);

    #pragma omp parallel num_threads(2)
    {
        int thread_num = omp_get_thread_num();
        cudaSetDevice(thread_num);

        cudaMalloc((void **)&d_dist[thread_num], n * n * sizeof(int));
        cudaMemcpy(d_dist[thread_num], Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

        phase1<<< 1, block2, 4096*sizeof(int) >>>(B, 0,    0,  0,  1,  1, n, d_dist[thread_num], n);
        phase2<<< grid2, block, 8192*sizeof(int) >>>(B, 0, n, d_dist[thread_num], n); 
        phase3<<<grid3, block2>>>(B, 0, n, d_dist[thread_num], n, thread_num, round);
        
        for (int r = 1; r < round; ++r) {

            #pragma omp barrier

            if (r <= (round/2) && thread_num == 1) {
                cudaMemcpyPeer(d_dist[1] + r * B * n, 1, d_dist[0] + r * B * n, 0, B * n * sizeof(int));
            } else if (r > (round/2) && thread_num == 0) {
                cudaMemcpyPeer(d_dist[0] + r * B * n, 0, d_dist[1] + r * B * n, 1, B * n * sizeof(int));
            }

            #pragma omp barrier

            phase1<<< 1, block2, 4096*sizeof(int) >>>(B, r,    r,  r,  1,  1, n, d_dist[thread_num], n);
            phase2<<< grid2, block, 8192*sizeof(int) >>>(B, r, n, d_dist[thread_num], n); 
            phase3<<<grid3, block2>>>(B, r, n, d_dist[thread_num], n, thread_num, round);
            
        }

        if (thread_num == 0)
            cudaMemcpy(Dist, d_dist[0], (round/2) * B * n * sizeof(int), cudaMemcpyDeviceToHost);
        else if (thread_num == 1)
            cudaMemcpy(&Dist[(round/2) * B * n], d_dist[1] + (round/2) * B * n, (n - (round/2) * B) * n * sizeof(int), cudaMemcpyDeviceToHost);
    }
}

__global__ 
void phase1(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* d_dist, int pitch_int) {

    int b_i = (block_start_x << 6) + threadIdx.y;
    int b_j = (block_start_y << 6) + threadIdx.x;
    
    extern __shared__ int shared_mem[]; 
    
    #pragma unroll
    for(int r=0; r<2; ++r){
        int idx = threadIdx.y + (r << 5);
        shared_mem[idx*B + threadIdx.x] = d_dist[(b_i + (r << 5))*n + b_j]; 
        shared_mem[idx*B + threadIdx.x + 32] = d_dist[(b_i + (r << 5))*n + b_j + 32];      
    }

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();

        for(int r=0; r<2; ++r){       
            int idx = threadIdx.y + (r << 5);
            shared_mem[idx*B+threadIdx.x] = min(shared_mem[idx*B+threadIdx.x], shared_mem[idx*B+k] + shared_mem[k*B+threadIdx.x]);
            shared_mem[idx*B+threadIdx.x + 32] = min(shared_mem[idx*B+threadIdx.x + 32], shared_mem[idx*B+k] + shared_mem[k*B+threadIdx.x + 32]);
            
        }
        
    }

    #pragma unroll
    for(int r=0; r<2; ++r){
        d_dist[(b_i + (r << 5))*n + b_j] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x];  
        d_dist[(b_i + (r << 5))*n + b_j + 32] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x + 32];  
    }
}

extern __shared__ int shared_mem[]; 
__global__ void phase2(int B, int Round, int n, int* d_dist, int pitch_int) {
    
    int b_i, b_j;
    if(blockIdx.y==0){
        b_i = Round;
        b_j = blockIdx.x + (blockIdx.x>=Round);
    }
    else{
        b_i = blockIdx.x + (blockIdx.x>=Round);
        b_j = Round;
    }

    b_j = b_j * B + threadIdx.x;
    
    #pragma unroll
    for(int r=0; r<4; r++){
        int idx = threadIdx.y + 16 * r;
        shared_mem[idx*B + threadIdx.x] = d_dist[(b_i * B + idx)*pitch_int + b_j];
        shared_mem[idx*B + threadIdx.x + B*B] = d_dist[(Round * B + idx)*pitch_int + Round * B + threadIdx.x];
    }

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        for(int r=0; r<4; r++){
            int idx = threadIdx.y + 16 * r;
            shared_mem[idx*B+threadIdx.x] = min(shared_mem[idx*B+threadIdx.x], shared_mem[idx*B+k + !blockIdx.y*B*B] + shared_mem[k*B+threadIdx.x + blockIdx.y*B*B]);
        }
    }
    #pragma unroll
    for(int r=0; r<4; r++){
        int idx = threadIdx.y + 16 * r;
        d_dist[(b_i * B + idx)*pitch_int + b_j] = shared_mem[idx*B + threadIdx.x];  
    }   
}

__global__ void phase3(int B, int Round, int n, int* d_dist, int pitch_int, int thread_num, int total_round) {

    int block_y = blockIdx.y + thread_num*(total_round/2);

    int k_min = Round*B;

    block_y = block_y + (block_y>=Round);

    if(block_y >= total_round) return;

    int b_i = (block_y << 6) + threadIdx.y;
    int b_j = ((blockIdx.x + (blockIdx.x>=Round)) << 6) + threadIdx.x;

    __shared__ int shared_mem[8192]; 

    int dist1 = d_dist[b_i*n + b_j];
    int dist2 = d_dist[b_i*n + b_j + 32];
    int dist3 = d_dist[(b_i+32)*n + b_j];
    int dist4 = d_dist[(b_i+32)*n + b_j + 32];
    
    #pragma unroll
    for(int r=0; r<2; ++r){
        int idx = threadIdx.y + ( r << 5);
        shared_mem[(idx << 6) + threadIdx.x ] = d_dist[(b_i + ( r << 5))*n + k_min + threadIdx.x];
        shared_mem[(idx << 6) + threadIdx.x + 32] = d_dist[(b_i + ( r << 5))*n + k_min + threadIdx.x + 32];
        shared_mem[(idx << 6) + threadIdx.x + 4096] = d_dist[(k_min + idx)*n + b_j];
        shared_mem[(idx << 6) + threadIdx.x + 4128] = d_dist[(k_min + idx)*n + b_j + 32];
        
    }

    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int idx = (threadIdx.y << 6) + k;
        int val1 = shared_mem[idx] + shared_mem[(k << 6) + threadIdx.x + 4096];
        int val2 = shared_mem[idx] + shared_mem[(k << 6) + threadIdx.x + 4128];
        int val3 = shared_mem[idx + 2048] + shared_mem[(k << 6) + threadIdx.x + 4096];
        int val4 = shared_mem[idx + 2048] + shared_mem[(k << 6) + threadIdx.x + 4128];
        dist1 = min(dist1, val1);
        dist2 = min(dist2, val2);
        dist3 = min(dist3, val3);
        dist4 = min(dist4, val4);
    }

    d_dist[b_i*n + b_j] = dist1;
    d_dist[b_i*n + b_j + 32] = dist2;
    d_dist[(b_i+32)*n + b_j] = dist3;
    d_dist[(b_i+32)*n + b_j + 32] = dist4;

}