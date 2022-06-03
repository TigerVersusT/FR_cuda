#include <vector>
#include <iostream>
#include <omp.h>
/* naive translation of original C++ code, since the only one block is used, there leaves
    losts of improve space, see utillsV2
 */

texture<float, 2> texRef;

__global__ void directionTop(const float *img, const float *weights,
                             const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[n + 2 - group - 1];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    {
        int pColumn = column + n * 2;
        int startRow = group; // k
        int startCol = n;     // n

        result[curRow * column + curCol] +=
            weight * tex2D(texRef, (curRow + startRow), curCol + startCol);
    }

    return;
}

__global__ void directionBottom(const float *img, const float *weights,
                                const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    {
        int pColumn = column + n * 2;
        int startRow = n - 1 + group;
        int startCol = n;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }
    return;
}

__global__ void directionLeft(const float *img, const float *weights,
                              const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[n + 2 - group - 1];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    {
        int pColumn = column + n * 2;
        int startRow = n;
        int startCol = group;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}
__global__ void directionRight(const float *img, const float *weights,
                               const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    {
        int pColumn = column + n * 2;
        int startRow = n;
        int startCol = n - 1 + group;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}
__global__ void directionLeftBottom(const float *img, const float *weights,
                                    const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    {
        int pColumn = column + n * 2;
        int startRow = n + group - 1;
        int startCol = n - group + 1;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}
__global__ void directionRightTop(const float *img, const float *weights,
                                  const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    { // int column = blockDim.x;
        int pColumn = column + n * 2;
        int startRow = n + 1 - group;
        int startCol = n - 1 + group;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}
__global__ void directionLeftTop(const float *img, const float *weights,
                                 const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = blockIdx.y * blockDim.y + threadIdx.y;
    int curCol = blockIdx.x * blockDim.x + threadIdx.x;

    if ((curRow < row) && (curCol < column))
    { // int column = blockDim.x;
        int pColumn = column + n * 2;
        int startRow = n + 1 - group;
        int startCol = n + 1 - group;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}
__global__ void directionRightBottom(const float *img, const float *weights,
                                     const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    // int curRow = threadIdx.y;
    // int curCol = threadIdx.x;
    int curRow = threadIdx.x;
    int curCol = threadIdx.y;

    if ((curRow < row) && (curCol < column))
    { // int column = blockDim.x;
        int pColumn = column + n * 2;
        int startRow = n - 1 + group;
        int startCol = n - 1 + group;

        result[curRow * column + curCol] +=
            weight * img[(curRow + startRow) * pColumn + curCol + startCol];
    }

    return;
}

/* since the thread block has shape 16X16, some remain elements should be calculated separately on cpu */
void dTopRemain(const int N, const int row, const int column,
                const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;

    for (int i = 0; i < N; i++)
    {
        float weight = weights[N - i - 1];
        int pColumn = column + 2 * n;
        int startRow = i;
        int startCol = n;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }

        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }

        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dBottomRemain(const int N, const int row, const int column,
                   const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n - 1 + i;
        int startCol = n;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dLeftRemain(const int N, const int row, const int column,
                 const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[N - i - 1];
        int pColumn = column + 2 * n;
        int startRow = n;
        int startCol = i;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dRightRemain(const int N, const int row, const int column,
                  const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n;
        int startCol = n - 1 + i;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dLeftBottomRemain(const int N, const int row, const int column,
                       const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n + i - 1;
        int startCol = n - i + 1;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dRightTopRemain(const int N, const int row, const int column,
                     const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n + 1 - i;
        int startCol = n - 1 + i;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dLeftTopRemain(const int N, const int row, const int column,
                    const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n + 1 - i;
        int startCol = n + 1 - i;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

void dRightBottomRemain(const int N, const int row, const int column,
                        const int rowRemain, const int colRemain, const float *img, const float *weights, float *result)
{
    if ((rowRemain == 0) && (colRemain == 0))
    {
        return;
    }

    int n = N - 2;
    for (int i = 0; i < N; i++)
    {
        float weight = weights[i];
        int pColumn = column + 2 * n;
        int startRow = n - 1 + i;
        int startCol = n - 1 + i;
        int endRow = startRow + row;
        int endCol = startCol + column;

        // deal with bottom left elements
        if (rowRemain != 0)
        {
            for (int r = endRow - rowRemain; r < endRow; r++)
            {
                for (int c = startCol; c < endCol - colRemain; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with right left elements
        if (colRemain != 0)
        {
            for (int r = startRow; r < endRow - rowRemain; r++)
            {
                for (int c = endCol - colRemain; c < endCol; c++)
                {
                    result[(r - startRow) * column + c - startCol] +=
                        weight * img[r * pColumn + c];
                }
            }
        }
        // deal with RightBottom left elements
        for (int r = endRow - rowRemain; r < endRow; r++)
        {
            for (int c = endCol - colRemain; c < endCol; c++)
            {
                result[(r - startRow) * column + c - startCol] +=
                    weight * img[r * pColumn + c];
            }
        }
    }
}

/* use cuda to accelrate Dl_power and D_ls_power */
__global__ void powerKernel(const float *img, float *result, float power)
{

    int i = blockIdx.x * 256 + threadIdx.x;

    if (power < 0)
    {
        result[i] = 1 / (pow(img[i], abs(power)));
    }
    else
    {
        result[i] = pow(img[i], power);
    }

    return;
}

void powerRemain(const float *img, float *result, float power, int remain, int size)
{
#pragma parallel for
    for (int i = size - remain; i < remain; i++)
    {
        if (power < 0)
        {
            result[i] = 1 / (pow(img[i], abs(power)));
        }
        else
        {
            result[i] = pow(img[i], power);
        }
    }
}

__global__ void secondTermKernel(const float *img, float *result, float alpha, float power)
{
    int i = blockIdx.x * 256 + threadIdx.x;

    if (power < 0)
    {
        result[i] = (alpha / (pow(abs(img[i]), abs(power)))) * img[i];
    }
    else
    {
        result[i] = alpha * pow(abs(img[i]), power) * img[i];
    }

    return;
}

void secondTermRemain(const float *img, float *result, float alpha, float power, int remain, int size)
{
#pragma omp parallel for
    for (int i = size - remain; i < remain; i++)
    {
        if (power < 0)
        {
            result[i] = (alpha / (pow(abs(img[i]), abs(power)))) * img[i];
        }
        else
        {
            result[i] = alpha * pow(abs(img[i]), power) * img[i];
        }
    }
}

__global__ void tempKernel(const float *D_l_power, const float *D_l_8, const float *D_ls_8, const float *second_term,
                           const float *D_ls_power, float alpha, float *temp)
{
    int i = blockIdx.x * 256 + threadIdx.x;

    temp[i] = D_l_power[i] * D_l_8[i] + second_term[i] +
              alpha * D_ls_power[i] * D_ls_8[i];

    return;
}

void tempRemain(const float *D_l_power, const float *D_l_8, const float *D_ls_8, const float *second_term,
                const float *D_ls_power, float alpha, float *temp, int size, int remain)
{
#pragma omp parallel for
    for (int i = size - remain; i < remain; i++)
    {
        temp[i] = D_l_power[i] * D_l_8[i] + second_term[i] +
                  alpha * D_ls_power[i] * D_ls_8[i];
    }

    return;
}