#include "kernels/utills.h"

/* use cuda to optimize array power operation, use multiple stream for acceleration */
void arrayPowerCuda(const float *img, float *result, const int size, const float power)
{
    float *img_d{nullptr}, *result_d{nullptr};
    size_t imgSize{size * sizeof(float)};
    printIfCudaFailed(cudaMalloc(&img_d, imgSize));
    printIfCudaFailed(cudaMemcpy(img_d, img, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMalloc(&result_d, imgSize));

    int threadsPerBlock{256};
    int numBlocks(size / 256);
    int remain = size % 256;
    powerKernel<<<numBlocks, threadsPerBlock>>>(img_d, result_d, power);
    printIfCudaFailed(cudaMemcpy(result, result_d, imgSize, cudaMemcpyDeviceToHost));
    powerRemain(img, result, power, remain, size);

    cudaFree(img_d);
    cudaFree(result_d);

    return;
}

/* use cuda to optimize array power operation, use multiple stream for acceleration */
void arrayPowerCudaAsync(const float *img, float *result, const int size, const float power)
{
    float *img_d{nullptr}, *result_d{nullptr};
    size_t imgSize{size * sizeof(float)};
    printIfCudaFailed(cudaMalloc(&img_d, imgSize));
    printIfCudaFailed(cudaMemcpyAsync(img_d, img, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMalloc(&result_d, imgSize));

    int itemPerStream{size / 8};
    cudaStream_t stream[8];
    for (int i = 0; i < 8; ++i)
        cudaStreamCreate(&stream[i]);
    for (int i = 0; i < 8; i++)
    {
        printIfCudaFailed(cudaMemcpyAsync(img_d + itemPerStream * i, img + itemPerStream * i,
                                          itemPerStream * sizeof(float), cudaMemcpyHostToDevice));
        int threadsPerBlock{256};
        int numBlocks(itemPerStream / 256);
        int remain = itemPerStream % 256;
        powerKernel<<<numBlocks, threadsPerBlock>>>(img_d + itemPerStream * i, result_d + itemPerStream * i, power);
        printIfCudaFailed(cudaMemcpyAsync(result, result_d, itemPerStream * sizeof(float), cudaMemcpyDeviceToHost));
        powerRemain(img + itemPerStream * i, result + itemPerStream * i, power, remain, itemPerStream);
    }

    cudaFree(img_d);
    cudaFree(result_d);

    return;
}