#include <iostream>

#include "gpu.hh"

#define HEIGHT_MAX 50
#define cudaCheckError() {                                                                \
        cudaError_t e=cudaGetLastError();                                                 \
        if(e!=cudaSuccess) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }                                                                                     \

__device__ inline int* at(int *addr, int x, int y, int pitch) {
    return (int*)(char*)addr + pitch * y + x * sizeof(int);
}

__global__ void relabel(int *excess, int *neighbors[4], int *heights, int width,
        int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (excess[idx] <= 0 || heights[idx] >= HEIGHT_MAX)
        return;
    const int x_nghb[4] = {0, 1, 0, -1}; // idx offset for x axis
    const int y_nghb[4] = {-1, 0, 1, 0}; // idx offset for y axis
    int tmp_height = HEIGHT_MAX;
    for (int i = 0; i < 4; i++)
    {
        int idx_nghb = (y + y_nghb[i]) * width + (x + x_nghb[i]);
        if (neighbors[i][idx_nghb] > 0)
            tmp_height = tmp_height > heights[idx_nghb] + 1 ? heights[idx_nghb] : tmp_height;
    }
    heights[idx] = tmp_height;

}

__global__ void push(int *excess, int *neighbors[4], int *heights, int width, 
        int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (excess[idx] <= 0 || heights[idx] >= HEIGHT_MAX)
        return;
    const int x_nghb[4] = {0, 1, 0, -1}; // idx offset for x axis
    const int y_nghb[4] = {-1, 0, 1, 0}; // idx offset for y axis
    const int id_opp[4] = {2, 3, 0, 1};

    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + y_nghb[i]) * width + (x + x_nghb[i]);
        if (y + y_nghb[i] < 0 || y + y_nghb[i] >= height ||
            x + x_nghb[i] < 0 || x + x_nghb[i] >= width)
            continue;
        if (heights[idx_nghb] != heights[idx] - 1)
            return;
        int flow = neighbors[i][idx] > excess[idx] ? excess[idx] : neighbors[i][idx];
        // make atomic here
        excess[idx] -= flow;
        excess[idx_nghb] += flow;

        neighbors[i][idx] -= flow;
        neighbors[id_opp[i]][idx_nghb] += flow;
    }
}

__global__ void print_value(int *array, int pitch, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
}


int *duplicate_on_gpu(int *vect, int width, int height, size_t pitch)
{
    int *array;

    cudaMallocPitch(&array, &pitch, width * sizeof(int), height);
    cudaCheckError();
    std::cout << "Pitch: "<<pitch<<std::endl;
    cudaMemcpy2D(array, pitch,
                 vect, width * sizeof(int),
                 width * sizeof(int), height, cudaMemcpyHostToDevice);
    cudaCheckError();
    return array;
}

void max_flow_gpu(Graph graph)
{
    // Setting dimension
    int width  = graph._width;
    int height = graph._height;
    int w = std::ceil((float)width / 32);
    int h = std::ceil((float)height / 32);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(w, h);


    // Allocate for gpu
    size_t pitch;
    int *excess = duplicate_on_gpu(graph._excess_flow, width, height, pitch);
    int *heights = duplicate_on_gpu(graph._heights, width, height, pitch);
    int *tmp_heights;
    int *up = duplicate_on_gpu(graph._neighbors[0], width, height, pitch);
    int *right = duplicate_on_gpu(graph._neighbors[1], width, height, pitch);
    int *bottom = duplicate_on_gpu(graph._neighbors[2], width, height, pitch);
    int *left = duplicate_on_gpu(graph._neighbors[3], width, height, pitch);


   /* while ()
    {
        relabel<<<dimGrid, dimBlock>>>();
        push<<<dimGrid, dimBlock>>>();
    }*/
}
