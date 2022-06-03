#include <vector>
#include <iostream>
#include "kernels/utills.h"
#include <omp.h>

using namespace std;

/* use cuda to accerate cal8Deriv */
void cal8DerivCuda(const vector<float> &vChanelImg, vector<vector<float>> &D8,
                   vector<float> &D, const vector<float> &mask, const int row,
                   const int column, const int N, const int norm = 1)
{
    // CAL8DERIV 计算八个方向上的分数阶偏导数
    // 根据系数向量，对输入进来的矩阵计算八个方向上的偏导数矩阵
    // C 的长度为 n + 2
    // D8 的第三个维度的排列顺序为 u, d, l, r, ld, ru, lu, rd
    int n = N - 2;

    int pRow = row + n * 2;
    int pColumn = column + n * 2;
    size_t result_size{row * column * sizeof(float)};
    size_t img_size{pRow * pColumn * sizeof(float)};

    /* prepare data */
    float *mask_d{nullptr};
    printIfCudaFailed(cudaMalloc(&mask_d, result_size));
    printIfCudaFailed(cudaMemcpy(mask_d, &(mask[0]), result_size, cudaMemcpyHostToDevice));

    float *vChanelImg_d{nullptr};
    printIfCudaFailed(cudaMalloc(&vChanelImg_d, img_size));
    printIfCudaFailed(cudaMemcpy(vChanelImg_d, &(vChanelImg[0]), img_size, cudaMemcpyHostToDevice));

    cudaStream_t stream[8];
    for (int i = 0; i < 8; ++i)
        cudaStreamCreate(&stream[i]);
    // 分别计算八个方向的分数阶偏导数矩阵
    dim3 threadsPerBlock(16, 16);
    int rowBlock{row / 16}, rowRemain{row % 16};
    if (rowRemain != 0)
    {
        rowBlock++;
    }
    int colBlock{column / 16}, colRemain{column % 16};
    if (colRemain != 0)
    {
        colBlock++;
    }
    // note that, this kind of block is not the usually used one, which put column as the first parameter
    dim3 numBlocks(rowBlock, colBlock);
    float *dTop{nullptr}, *dBottom{nullptr}, *dLeft{nullptr}, *dRight{nullptr}, *dLeftBottom{nullptr},
        *dRightTop{nullptr}, *dLeftTop{nullptr}, *dRightBottom{nullptr};

    omp_set_num_threads(8);
#pragma omp parallel sections
    {
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dTop, result_size));
            for (int i = 0; i < N; i++)
            {
                directionTop<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dTop);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[0][0]), dTop, result_size, cudaMemcpyDeviceToHost, stream[0]));
        }

#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dBottom, result_size));
            for (int i = 0; i < N; i++)
            {
                directionBottom<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dBottom);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[1][0]), dBottom, result_size, cudaMemcpyDeviceToHost, stream[1]));
        }
#pragma omp section
        {
            auto idx = omp_get_thread_num();

            printIfCudaFailed(cudaMalloc(&dLeft, result_size));
            for (int i = 0; i < N; i++)
            {
                directionLeft<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dLeft);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[2][0]), dLeft, result_size, cudaMemcpyDeviceToHost, stream[2]));
        }
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dRight, result_size));
            for (int i = 0; i < N; i++)
            {
                directionRight<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dRight);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[3][0]), dRight, result_size, cudaMemcpyDeviceToHost, stream[3]));
        }
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dLeftBottom, result_size));
            for (int i = 0; i < N; i++)
            {
                directionLeftBottom<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dLeftBottom);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[4][0]), dLeftBottom, result_size, cudaMemcpyDeviceToHost, stream[4]));
        }
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dRightTop, result_size));
            for (int i = 0; i < N; i++)
            {
                directionRightTop<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dRightTop);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[5][0]), dRightTop, result_size, cudaMemcpyDeviceToHost, stream[5]));
        }
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dLeftTop, result_size));
            for (int i = 0; i < N; i++)
            {
                directionLeftTop<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dLeftTop);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[6][0]), dLeftTop, result_size, cudaMemcpyDeviceToHost, stream[6]));
        }
#pragma omp section
        {
            printIfCudaFailed(cudaMalloc(&dRightBottom, result_size));
            for (int i = 0; i < N; i++)
            {
                directionRightBottom<<<numBlocks, threadsPerBlock>>>(vChanelImg_d, mask_d, N - 2, i, column, row, dRightBottom);
            }

            printIfCudaFailed(cudaMemcpyAsync(&(D8[7][0]), dRightBottom, result_size, cudaMemcpyDeviceToHost, stream[7]));
        }
    }
    /*
    #pragma omp parallel sections
    {
    #pragma omp section
        {
            dTopRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[0][0]));
        }
    #pragma omp section
        {
            dBottomRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[1][0]));
        }
    #pragma omp section
        {

            dLeftRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[2][0]));
        }
    #pragma omp section
        {
            dRightRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[3][0]));
        }
    #pragma omp section
        {

            dLeftBottomRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[4][0]));
        }
    #pragma omp section
        {
            dRightTopRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[5][0]));
        }
    #pragma omp section
        {

            dLeftTopRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[6][0]));
        }
    #pragma omp section
        {

            dRightBottomRemain(N, row, column, rowRemain, colRemain, &(vChanelImg[0]), &(mask[0]), &(D8[7][0]));
        }
    }
    */
    // 计算全微分
    if (norm == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < row * column; i++)
        {
            D[i] += abs(D8[0][i]) + abs(D8[1][i]) + abs(D8[2][i]) + abs(D8[3][i]) +
                    abs(D8[4][i]) + abs(D8[5][i]) + abs(D8[6][i]) +
                    abs(D8[7][i]);
        }
    }

    /* free all memorys */
    cudaFree(mask_d);
    cudaFree(vChanelImg_d);
    cudaFree(dTop);
    cudaFree(dBottom);
    cudaFree(dLeft);
    cudaFree(dRight);
    cudaFree(dLeftBottom);
    cudaFree(dRightTop);
    cudaFree(dLeftTop);
    cudaFree(dRightBottom);

    return;
}
