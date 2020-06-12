#include <iostream>

#include "gpu.hh"

#define HEIGHT_MAX 1000
#define cudaCheckError() {                                                                \
        cudaError_t e=cudaGetLastError();                                                 \
        if(e!=cudaSuccess) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }                                                                                     \

__constant__ int x_nghb[4] = {0, 1, 0, -1}; // idx offset for x axis
__constant__ int y_nghb[4] = {-1, 0, 1, 0}; // idx offset for y axis
__constant__ int id_opp[4] = {2, 3, 0, 1};

__device__ inline int* at(int *addr, int x, int y, int pitch) {
    return (int*)((char*)addr + pitch * y + x * sizeof(int));
}

__global__ void relabel(int *excess, int *heights, int *tmp_heights, int *up,
        int *right, int *bottom, int *left, int width, int height, size_t pitch,
        int height_max)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (*at(excess, x, y, pitch) <= 0 || *at(heights, x, y, pitch) >= height_max)
        return;

    int *neighbors[4] = {up, right, bottom, left};
    int tmp_height = height_max;

    for (int i = 0; i < 4; i++)
    {
        int new_x = x + x_nghb[i];
        int new_y = y + y_nghb[i];
        if (*at(neighbors[i], x, y, pitch) > 0)
            tmp_height = min(tmp_height, *at(heights, new_x, new_y, pitch) + 1);
    }
    *at(tmp_heights, x, y, pitch) = tmp_height;
}

__global__ void push(int *excess, int *heights, int *up, int *right,
        int *bottom, int *left, int width, int height, int pitch, int height_max)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (*at(excess, x, y, pitch) <= 0 || *at(heights, x, y, pitch) >= height_max)
        return;

    int *neighbors[4] = {up, right, bottom, left};

    for (auto i = 0; i < 4; i++) {
        int new_x = x + x_nghb[i];
        int new_y = y + y_nghb[i];

        if (new_y < 0 || new_y >= height || new_x < 0 || new_x >= width)
            continue;

        if (*at(heights, new_x, new_y, pitch) != *at(heights, x, y, pitch) - 1)
            continue;

        int flow = min(*at(neighbors[i], x, y, pitch), *at(excess, x, y, pitch));

        atomicAdd(at(excess, x, y, pitch),        -flow);
        atomicAdd(at(excess, new_x, new_y, pitch), flow);

        atomicAdd(at(neighbors[i], x, y, pitch),                -flow);
        atomicAdd(at(neighbors[id_opp[i]], new_x, new_y, pitch), flow);
    }
}


int *duplicate_on_gpu(int *vect, int width, int height, size_t &pitch)
{
    int *array;

    cudaMallocPitch(&array, &pitch, width * sizeof(int), height);
    cudaCheckError();
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
    int size = graph._height_max;
    //int size = 100;
    int w = std::ceil((float)width / 32);
    int h = std::ceil((float)height / 32);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(w, h);


    // Allocate for gpu
    size_t pitch;
    int *excess = duplicate_on_gpu(graph._excess_flow, width, height, pitch);
    int *heights = duplicate_on_gpu(graph._heights, width, height, pitch);
    int *tmp_heights = duplicate_on_gpu(graph._heights, width, height, pitch);
    int *up = duplicate_on_gpu(graph._neighbors[0], width, height, pitch);
    int *right = duplicate_on_gpu(graph._neighbors[1], width, height, pitch);
    int *bottom = duplicate_on_gpu(graph._neighbors[2], width, height, pitch);
    int *left = duplicate_on_gpu(graph._neighbors[3], width, height, pitch);

    while (graph.any_active())
    {
        // Double buffering not smart here
        cudaMemcpy2D(tmp_heights, pitch, heights, pitch, 
                 width * sizeof(int), height, cudaMemcpyDeviceToDevice);
        relabel<<<dimGrid, dimBlock>>>(excess, heights, tmp_heights, up, right, bottom, left,
            width, height, pitch, size);
        cudaMemcpy2D(heights, pitch, tmp_heights, pitch, 
                 width * sizeof(int), height, cudaMemcpyDeviceToDevice);

        // Push call
        push<<<dimGrid, dimBlock>>>(excess, heights, up, right, bottom, left,
            width, height, pitch, size);

        // Set back to cpu memory for any_active check not smart
        cudaMemcpy2D(graph._excess_flow, width * sizeof(int),
                     excess, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(graph._heights, width * sizeof(int),
                     heights, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
    }
    
    // recopying data from gpu to cpu for min cut
    cudaMemcpy2D(graph._neighbors[0], width * sizeof(int),
                     up, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(graph._neighbors[1], width * sizeof(int),
                     right, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(graph._neighbors[2], width * sizeof(int),
                     bottom, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(graph._neighbors[3], width * sizeof(int),
                     left, pitch,
                     width * sizeof(int), height, cudaMemcpyDeviceToHost);
}
