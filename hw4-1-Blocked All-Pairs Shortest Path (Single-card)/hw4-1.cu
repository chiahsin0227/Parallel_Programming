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
__global__ void phase1(int B, int Round, int block_start_x, int block_start_y, int n, int* d_dist);
__global__ void phase2_1(int B, int Round, int n, int* d_dist, int block_start_x, int block_start_y);
__global__ void phase2_2(int B, int Round, int n, int* d_dist, int block_start_x, int block_start_y);
__global__ void phase3(int B, int Round, int n, int* d_dist, int block_start_x, int block_start_y);

int n, m, v;   // Number of vertices, edges
int* Dist;
int* d_dist;

int main(int argc, char* argv[])
{   
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(d_dist);
    return 0;
}

inline void input(char* infile) { 
    FILE* file = fopen(infile, "rb"); 
    fread(&v, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    if(v%64) n = v + (64 - v%64);
    else n = v;

    cudaMallocHost( &Dist, sizeof(int)*(n*n)); 
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
    int round = ceil(n, B);cudaMalloc((void **)&d_dist, n * n * sizeof(int));
    cudaMemcpy(d_dist , Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    for (int r = 0; r < round; ++r) {

        int k_min = r << 6;
        /* Phase 1*/
        phase1<<< 1, block, B*B*sizeof(int) >>>(B, r,    r,  r, n, d_dist);

        /* Phase 2*/
        phase2_2<<<dim3(r, 1),             block, 8192*sizeof(int)>>>(B, r, n, d_dist, r, 0);  // up
        phase2_2<<<dim3(round - r - 1, 1), block, 8192*sizeof(int)>>>(B, r, n, d_dist, r, r+1);  // down
        phase2_1<<<dim3(1, r),             block, 8192*sizeof(int)>>>(B, r, n, d_dist, 0, r); // left
        phase2_1<<<dim3(1, round - r - 1), block, 8192*sizeof(int)>>>(B, r, n, d_dist, r+1, r); // right

        /* Phase 3*/
        phase3<<<dim3(r, r), block, 8192*sizeof(int)>>>(B, k_min, n, d_dist, 0, 0);
        phase3<<<dim3(round - r - 1, r), block, 8192*sizeof(int)>>>(B, k_min, n, d_dist, 0, r+1);
        phase3<<<dim3(r, round - r - 1), block, 8192*sizeof(int)>>>(B, k_min, n, d_dist, r+1, 0);
        phase3<<<dim3(round - r - 1, round - r - 1), block, 8192*sizeof(int)>>>(B, k_min, n, d_dist, r+1, r+1);
        
    }
    cudaMemcpy(Dist , d_dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ 
void phase1(int B, int Round, int block_start_x, int block_start_y, int n, int* d_dist) {

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

__global__ void phase2_1(int B, int Round, int n, int* d_dist, int block_start_x, int block_start_y) {

    int b_i = ((blockIdx.y + block_start_x) << 6) + threadIdx.y;
    int b_j = ((blockIdx.x + block_start_y) << 6) + threadIdx.x;
    
    extern __shared__ int shared_mem[]; 
    
    #pragma unroll
    for(int r=0; r<2; ++r){
        int idx = threadIdx.y + (r << 5);
        shared_mem[idx*B + threadIdx.x] = d_dist[(b_i + (r << 5))*n + b_j];    // ij = ik
        shared_mem[idx*B + threadIdx.x + 32] = d_dist[(b_i + (r << 5))*n + b_j + 32];    // ij = ik
        shared_mem[idx*B + threadIdx.x + 4096] = d_dist[((Round << 6) + idx)*n + b_j];  // kj
        shared_mem[idx*B + threadIdx.x + 4096 + 32] = d_dist[((Round << 6) + idx)*n + b_j + 32];  // kj
    }
    
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        for(int r=0; r<2; ++r){       
            int idx = threadIdx.y + (r << 5);
            shared_mem[idx*B+threadIdx.x] = min(shared_mem[idx*B+threadIdx.x], shared_mem[idx*B+k] + shared_mem[k*B+threadIdx.x + 4096]);
            shared_mem[idx*B+threadIdx.x + 32] = min(shared_mem[idx*B+threadIdx.x + 32], shared_mem[idx*B+k] + shared_mem[k*B+threadIdx.x + 4096 + 32]);

        }
        
    }

    #pragma unroll
    for(int r=0; r<2; ++r){
        d_dist[(b_i + (r << 5))*n + b_j] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x];  
        d_dist[(b_i + (r << 5))*n + b_j + 32] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x + 32];  
    }
    
}

__global__ 
void phase2_2(int B, int Round, int n, int* d_dist, int block_start_x, int block_start_y) {

    int b_i = ((blockIdx.y + block_start_x) << 6) + threadIdx.y;
    int b_j = ((blockIdx.x + block_start_y) << 6) + threadIdx.x;
    
    extern __shared__ int shared_mem[]; 
    
    #pragma unroll
    for(int r=0; r<2; ++r){
        int idx = threadIdx.y + (r << 5);
        shared_mem[idx*B + threadIdx.x] = d_dist[(b_i + (r << 5))*n + b_j];    // ij = kj
        shared_mem[idx*B + threadIdx.x + 32] = d_dist[(b_i + (r << 5))*n + b_j + 32];    // ij = kj
        shared_mem[idx*B + threadIdx.x + 4096] = d_dist[(b_i + ( r << 5))*n + (Round << 6) + threadIdx.x];  // ik
        shared_mem[idx*B + threadIdx.x + 4096 + 32] = d_dist[(b_i + ( r << 5))*n + (Round << 6) + threadIdx.x + 32];  // ik
    }

    

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        for(int r=0; r<2; ++r){       
            int idx = threadIdx.y + (r << 5);
            shared_mem[idx*B+threadIdx.x] = min(shared_mem[idx*B+threadIdx.x], shared_mem[idx*B+k+4096] + shared_mem[k*B+threadIdx.x]);
            shared_mem[idx*B+threadIdx.x + 32] = min(shared_mem[idx*B+threadIdx.x + 32], shared_mem[idx*B+k+4096] + shared_mem[k*B+threadIdx.x + 32]);
            
        }
        
    }

    #pragma unroll
    for(int r=0; r<2; ++r){
        d_dist[(b_i + (r << 5))*n + b_j] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x];  
        d_dist[(b_i + (r << 5))*n + b_j + 32] = shared_mem[(threadIdx.y + (r << 5))*B + threadIdx.x + 32];  
    }
    
}

__global__ void phase3(int B, int k_min, int n, int* d_dist, int block_start_x, int block_start_y) {

    int b_i = ((blockIdx.y + block_start_x) << 6) + threadIdx.y;
    int b_j = ((blockIdx.x + block_start_y) << 6) + threadIdx.x;

    extern __shared__ int shared_mem[];

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