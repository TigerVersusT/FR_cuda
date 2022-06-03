#include "kernels/utills.h"

/* use cuda to optimize calculation of second term, use multi-stream for further accelaration */
void tempCuda(const float *D_l_power, const float *D_l_8, const float *second_term,
              const float *D_ls_power, const float *D_ls_8, float alpha, float *temp, int size)
{
    size_t imgSize{size * sizeof(float)};
    float *D_l_power_d{nullptr}, *D_l_8_d{nullptr}, *second_term_d{nullptr},
        *D_ls_power_d{nullptr}, *D_ls_8_d{nullptr}, *temp_d{nullptr};

    printIfCudaFailed(cudaMalloc(&D_l_power_d, imgSize));
    printIfCudaFailed(cudaMalloc(&D_l_8_d, imgSize));
    printIfCudaFailed(cudaMalloc(&D_ls_8_d, imgSize));
    printIfCudaFailed(cudaMalloc(&D_ls_power_d, imgSize));
    printIfCudaFailed(cudaMalloc(&second_term_d, imgSize));
    printIfCudaFailed(cudaMalloc(&temp_d, imgSize));

    printIfCudaFailed(cudaMemcpy(D_l_power_d, D_l_power, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(D_l_8_d, D_l_8, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(D_ls_8_d, D_ls_8, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(D_ls_power_d, D_ls_power, imgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(second_term_d, second_term, imgSize, cudaMemcpyHostToDevice));

    int threadsPerBlock{256};
    int numBlocks(size / 256);
    int remain = size % 256;
    tempKernel<<<numBlocks, threadsPerBlock>>>(D_l_power_d, D_l_8_d, D_ls_8_d, second_term_d,
                                               D_ls_power_d, alpha, temp_d);
    tempRemain(D_l_power, D_l_8, D_ls_8, second_term,
               D_ls_power, alpha, temp, size, remain);

    cudaFree(D_l_power_d);
    cudaFree(D_l_8_d);
    cudaFree(D_ls_8_d);
    cudaFree(D_ls_power_d);
    cudaFree(second_term_d);
    cudaFree(temp_d);

    return;
}