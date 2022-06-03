#include <vector>
#include <iostream>
#include "kernels/utills.h"

using namespace std;

/*
    cuda optimized version of fstDiff, note that the given images should be pad bofre calling
    this function.
*/

void fstDiffCuda(const vector<float> &A_pad, vector<float> &result, const int direction,
                 const int row, const int column)
{
    // [0, 1, 2, 3, 4, 5, 6, 7] := [D_u, D_d, D_l, D_r, D_ld, D_ru, D_lu, D_rd]

    int N{4};
    vector<float> mask{0.375, 0.375, -0.875, 0.125};
    float *mask_d{nullptr};
    printIfCudaFailed(cudaMalloc(&mask_d, sizeof(float) * N));
    printIfCudaFailed(cudaMemcpy(mask_d, &(mask[0]), sizeof(float) * N, cudaMemcpyHostToDevice));

    int n = 2;

    float *result_d{nullptr};
    int imgSize = (row + n * 2) * (column + n * 2);
    int resultSize = row * column;

    cudaChannelFormatDesc desc;
    desc = cudaCreateChannelDesc<float>();
    cudaArray *array = nullptr;
    cudaMallocArray(&array, &desc, row, column);
    cudaMemcpyToArray(array, 0, 0, &(A_pad[0]), sizeof(float) * resultSize, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(texRef, array);

    printIfCudaFailed(cudaMalloc(&result_d, sizeof(float) * resultSize));

    dim3 threadsPerBlock(16, 16);
    int rowBlock{row / 16}, rowRemain{row % 16};
    if (rowRemain != 0)
    {
        rowBlock++;
    }
    int colBlock{column / 16}, colRemain{column % 16};
    if (colRemain != 0)
    {
        colRemain++;
    }
    dim3 numBlocks(rowBlock, colBlock);
    switch (direction)
    {
    case 0:
        for (int i = 0; i < 4; i++)
        {
            directionTop<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));

        // dTopRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 1:
        for (int i = 0; i < 4; i++)
        {
            directionBottom<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dBottomRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 2:
        for (int i = 0; i < 4; i++)
        {
            directionLeft<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dLeftRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 3:
        for (int i = 0; i < 4; i++)
        {
            directionRight<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dRightRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 4:
        for (int i = 0; i < 4; i++)
        {
            directionLeftBottom<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dLeftBottomRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 5:
        for (int i = 0; i < 4; i++)
        {
            directionRightTop<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dRightTopRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 6:
        for (int i = 0; i < 4; i++)
        {
            directionLeftTop<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dLeftTopRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    case 7:
        for (int i = 0; i < 4; i++)
        {
            directionRightBottom<<<numBlocks, threadsPerBlock>>>(mask_d, N - 2, i, column, row, result_d);
        }
        printIfCudaFailed(cudaMemcpy(&(result[0]), result_d, sizeof(float) * resultSize, cudaMemcpyDeviceToHost));
        // dRightBottomRemain(N, row, column, rowRemain, colRemain, &(A_pad[0]), &(mask[0]), &(result[0]));

        break;
    default:
        break;
    }

    cudaUnbindTexture(texRef);
    cudaFreeArray(array);

    cudaFree(mask_d);
    cudaFree(result_d);

    return;
}
