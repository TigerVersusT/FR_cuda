#include "../utills.h"
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

// use two dimensional grid and two dimensional block
__global__ void update_l_curr(float *l_curr, float *delta_l, const int row,
                              const int column, const int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < column && r < row)
    {
        int idx1 = r * column + c;
        int idx2 = (r + n) * (column + 2 * n) + c + n;
        l_curr[idx2] += delta_l[idx1];
    }
}

// use one dimensional grid and one dimensional block, exist warp divergence
__global__ void imageRelpaceWith(float *img, const float threshold, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        if (img[x] < threshold)
        {
            img[x] = threshold;
        }
    }
}

// use one dimensional grid and one dimensional block
__global__ void imageExp(float *img, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        img[x] = expf(img[x]);
    }
}

// use one dimensional grid and one dimensional block
__global__ void imageSub(float *result, float *img1, float *img2, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        result[x] = img1[x] - img2[x];
    }
}

// use two dimensional grid and two dimensional block
__global__ void directionTop(const float *img, const float *weights,
                             const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[n + 2 - group - 1];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + group;
    int paddedC = c + n;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}

// use two dimensional grid and two dimensional block
__global__ void directionBottom(const float *img, const float *weights,
                                const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n - 1 + group;
    int paddedC = c + n;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}

// use two dimensional grid and two dimensional block
__global__ void directionLeft(const float *img, const float *weights,
                              const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[n + 2 - group - 1];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n;
    int paddedC = c + group;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}

// use two dimensional grid and two dimensional block
__global__ void directionRight(const float *img, const float *weights,
                               const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n;
    int paddedC = c + n - 1 + group;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}
__global__ void directionLeftBottom(const float *img, const float *weights,
                                    const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n + group - 1;
    int paddedC = c + n - group + 1;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}
__global__ void directionRightTop(const float *img, const float *weights,
                                  const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n + 1 - group;
    int paddedC = c + n - 1 + group;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}
__global__ void directionLeftTop(const float *img, const float *weights,
                                 const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n + 1 - group;
    int paddedC = c + n + 1 - group;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}
__global__ void directionRightBottom(const float *img, const float *weights,
                                     const int n, const int group, const int column, const int row, float *result)
{
    float weight = weights[group];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedR = r + n - 1 + group;
    int paddedC = c + n - 1 + group;
    int pColumn = column + n * 2;

    if (c < column && r < row)
    {
        result[r * column + c] +=
            weight * img[paddedR * pColumn + paddedC];
    }

    return;
}

// use one dimensional grid and one dimensional block
__global__ void computeD_l(float *d_D_l, float *d_D_l_8, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size)
    {
        float *base = d_D_l_8;
        float *base1 = base + size;
        float *base2 = base1 + size;
        float *base3 = base2 + size;
        float *base4 = base3 + size;
        float *base5 = base4 + size;
        float *base6 = base5 + size;
        float *base7 = base6 + size;

        d_D_l[x] = abs(*(base + x)) + abs(*(base1 + x)) +
                   abs(*(base2 + x)) + abs(*(base3 + x)) + abs(*(base4 + x)) +
                   abs(*(base5 + x)) + abs(*(base6 + x)) + abs(*(base7 + x));
    }
}

void cal8DerivCuda(const float *d_vchannelImg, float *d_D, float *d_D8,
                   const float *d_mask, const int row, const int column)
{
    int size = row * column;
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((column + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (row + threadsPerBlock.y - 1) / threadsPerBlock.y);

    omp_set_num_threads(8);
#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionTop<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                             i, column, row, d_D8);
            }
        }

#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionBottom<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                                i, column, row, d_D8 + size * 1);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionLeft<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                              i, column, row, d_D8 + size * 2);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionRight<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                               i, column, row, d_D8 + size * 3);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionLeftBottom<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                                    i, column, row, d_D8 + size * 4);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionRightTop<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                                  i, column, row, d_D8 + size * 5);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionLeftTop<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                                 i, column, row, d_D8 + size * 6);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < opt_.N; i++)
            {
                directionRightBottom<<<numBlocks, threadsPerBlock>>>(d_vchannelImg, d_mask, opt_.N - 2,
                                                                     i, column, row, d_D8 + size * 7);
            }
        }
    }

    // compute D_l
    dim3 thread{64, 1};
    dim3 block{(size + thread.x - 1) / thread.x, 1};
    computeD_l<<<block, thread>>>(d_D, d_D8, size);
}

// use one dimensional grid and one dimensional block
__global__ void arrayPower(const float *img, float *result, const float power, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > size)
    {
        return;
    }

    if (power < 0)
    {
        result[x] = 1 / (pow(img[x], abs(power)));
    }
    else
    {
        result[x] = pow(img[x], power);
    }
}

// use two dimensional grid and two dimensional block
__global__ void secondTerm(const float *img, float *result, const float power,
                           const float alpha, const int row, const int column, const int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r > row || c > column)
    {
        return;
    }

    int idx1 = r * column + c;
    int idx2 = (r + n) * (column + 2 * n) + c + n;
    if (power < 0)
    {
        result[idx1] = (alpha / (pow(abs(img[idx2]), abs(power)))) * img[idx2];
    }
    else
    {
        result[idx1] = alpha * pow(abs(img[idx2]), power) * img[idx2];
    }
}

// use two dimensional grid and two dimensional block
__global__ void computeTemp(const float *D_l_power, const float *D_l_8, const float *D_ls_power,
                            const float *D_ls_8, const float *second_term, const float alpha, const int row,
                            const int column, const int n, float *result)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if ((c < column) && (r < row))
    {
        int idx1 = r * column + c;
        int idx2 = (r + n) * (column + 2 * n) + c + n;

        result[idx2] = D_l_power[idx1] * D_l_8[idx1] + second_term[idx1] +
                       alpha * D_ls_power[idx1] * D_ls_8[idx1];
    }
}

// use one dimensional grid and one dimensional block
__global__ void addEightTerms(float *eight_terms, const float *temp, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        eight_terms[x] += temp[x];
    }
}

// use one dimensional grid and one dimensional block
__global__ void computeSumk(float *sumk, float *eight_terms,
                            const float factor, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        sumk[x] += factor * eight_terms[x];
    }
}

// use one dimensional grid and one dimensional block
__global__ void computeResults(float *results, float *sumk, const float factor, const int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        results[x] = factor * sumk[x];
    }
}

// use one dimensional grid and one dimensional block
__global__ void horizontalPad(float *img, const int row, const int column, const int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int paddedColumn = column + 2 * n;
    if (r < row)
    {
        // left
        for (int c = 0; c < n; c++)
        {
            int idx1 = (r + n) * paddedColumn + n - 1 - c;
            int idx2 = (r + n) * paddedColumn + n + c;
            img[idx1] = img[idx2];
        }

        // right
        for (int c = 0; c < n; c++)
        {
            int idx1 = (r + n) * paddedColumn + c + n + column;
            int idx2 = (r + n) * paddedColumn + n + column - 1 - c;
            img[idx1] = img[idx2];
        }
    }
}

// use one dimensional grid and one dimensional block
__global__ void verticalPad(float *img, const int row, const int column, const int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int paddedColumn = column + 2 * n;

    if (c < paddedColumn)
    {
        // top
        for (int r = 0; r < n; r++)
        {
            int idx1 = (n - 1 - r) * paddedColumn + c;
            int idx2 = (n + r) * paddedColumn + c;

            img[idx1] = img[idx2];
        }

        // bottom

        for (int r = 0; r < n; r++)
        {
            int idx1 = (n + row + r) * paddedColumn + c;
            int idx2 = (n + row - 1 - r) * paddedColumn + c;
            img[idx1] = img[idx2];
        }
    }
}

// note that the image itself is large enough to hold the padded imge
void symetricPad(float *img, const int row, const int column, const int n)
{

    float *d_img;
    printIfCudaFailed(cudaMalloc(&d_img, sizeof(float) * (row + 2 * n) * (column + 2 * n)));
    printIfCudaFailed(cudaMemcpy(d_img, img, sizeof(float) * (row + 2 * n) * (column + 2 * n),
                                 cudaMemcpyHostToDevice));

    dim3 threadBlock{64, 1};
    dim3 blockGrid{(row + threadBlock.x - 1) / threadBlock.x, 1};
    horizontalPad<<<blockGrid, threadBlock>>>(d_img, row, column, n);

    dim3 blockGrid2{(column + 2 * n + threadBlock.x - 1) / threadBlock.x, 1};
    verticalPad<<<threadBlock, blockGrid2>>>(d_img, row, column, n);

    printIfCudaFailed(cudaMemcpy(img, d_img, sizeof(float) * (row + 2 * n) * (column + 2 * n),
                                 cudaMemcpyDeviceToHost));
    printIfCudaFailed(cudaFree(d_img));
}

// note that the image itself is large enough to hold the padded imge
void symetricPadDevice(float *d_img, const int row, const int column, const int n)
{
    dim3 threadBlock{64, 1};
    dim3 blockGrid{(row + threadBlock.x - 1) / threadBlock.x, 1};
    horizontalPad<<<blockGrid, threadBlock>>>(d_img, row, column, n);

    dim3 blockGrid2{(column + 2 * n + threadBlock.x - 1) / threadBlock.x, 1};
    verticalPad<<<threadBlock, blockGrid2>>>(d_img, row, column, n);
}

void fstDiffCuda(float *d_img, float *d_result, const float *d_mask, const int layer,
                 const int row, const int column)
{
    int n = opt_.N - 2;
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((column + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (row + threadsPerBlock.y - 1) / threadsPerBlock.y);
    switch (layer)
    {
    case 0:
        for (int i = 0; i < 4; i++)
        {
            directionTop<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 1:
        for (int i = 0; i < 4; i++)
        {
            directionBottom<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 2:
        for (int i = 0; i < 4; i++)
        {
            directionLeft<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 3:
        for (int i = 0; i < 4; i++)
        {
            directionRight<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 4:
        for (int i = 0; i < 4; i++)
        {
            directionLeftBottom<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 5:
        for (int i = 0; i < 4; i++)
        {
            directionRightTop<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 6:
        for (int i = 0; i < 4; i++)
        {
            directionLeftTop<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    case 7:
        for (int i = 0; i < 4; i++)
        {
            directionRightBottom<<<numBlocks, threadsPerBlock>>>(d_img, d_mask, n, i, column, row, d_result);
        }

        break;
    default:
        break;
    }
}

void utTestPrintDeviceVector(float *d_img, string infor, int size, int limit, int column)
{
    vector<float> img(size, 0);
    printIfCudaFailed(cudaMemcpy(&(img[0]), d_img, sizeof(float) * size, cudaMemcpyDeviceToHost));

    cout << infor << endl;
    for (int i = 0; i < limit; i++)
    {
        cout << img[i]
             << "\t";
        if ((i + 1) % column == 0)
        {
            cout << "\n";
        }
    }
    cout << endl;
}

void computeP(float *d_l_curr, float *d_s, float *d_mask, float *d_ls, float *d_results,
              float *d_D_l, float *d_D_ls, float *d_D_l_8, float *d_D_ls_8,
              float *d_D_l_power, float *d_D_ls_power, float *d_eight_terms,
              float *d_second_term, float *d_temp_pad, float *d_temp_results,
              float *d_sumk, const int row, const int column)
{
    int n = opt_.N - 2;
    int size = row * column;
    int paddedSize = (row + 2 * n) * (column + 2 * n);

    dim3 threadBlock{64, 1};
    dim3 blockGrid{(paddedSize + threadBlock.x - 1) / threadBlock.x, 1};
    imageSub<<<blockGrid, threadBlock>>>(d_ls, d_l_curr, d_s, paddedSize);

    // iterativePad(l, l_pad, n, row, column);
    // symetricPadDevice(d_l_curr, row, column, n);

    // utTestPrintDeviceVector(d_l_curr, "d_l_curr", paddedSize, paddedSize, column + 2 * n);

    cudaMemset(d_D_l_8, 0, 8 * size * sizeof(float));
    cudaMemset(d_D_ls_8, 0, 8 * size * sizeof(float));
    cal8DerivCuda(d_l_curr, d_D_l, d_D_l_8, d_mask, row, column);

    // utTestPrintDeviceVector(d_D_l_8, "d_D_l_8 0", size, size, column);
    // utTestPrintDeviceVector(d_D_l_8 + size, "d_D_l_8 1", size, size, column);
    // utTestPrintDeviceVector(d_D_l_8 + size * 2, "d_D_l_8 2", size, size, column);
    // utTestPrintDeviceVector(d_D_l_8 + size * 3, "d_D_l_8 3", size, size, column);
    // utTestPrintDeviceVector(d_D_l_8 + size * 4, "d_D_l_8 4", size, size, column);
    // utTestPrintDeviceVector(d_D_l_8 + size * 5, "d_D_l_8 5", size, size, column);
    //  utTestPrintDeviceVector(d_D_l_8 + size * 6, "d_D_l_8 6", size, size, column);
    //   utTestPrintDeviceVector(d_D_l_8 + size * 7, "d_D_l_8 7", size, size, column);

    //  iterativePad(ls, ls_pad, n, row, column);
    // symetricPadDevice(d_ls, row, column, n);
    cal8DerivCuda(d_ls, d_D_ls, d_D_ls_8, d_mask, row, column);

    dim3 threadBlock1{64, 1};
    dim3 blockGrid1{(size + threadBlock1.x - 1) / threadBlock1.x, 1};
    imageRelpaceWith<<<blockGrid1, threadBlock1>>>(d_D_l, size, opt_.epsilon_1);
    // utTestPrintVector(D_l, "D_l");
    imageRelpaceWith<<<blockGrid1, threadBlock1>>>(d_D_ls, size, opt_.epsilon_1);
    // utTestPrintVector(D_ls, "D_ls");

    // utTestPrintDeviceVector(d_D_l, "d_D_l", size, size, column);
    // utTestPrintDeviceVector(d_D_ls, "d_D_ls", size, size, column);

    // 对 k 进行遍历，将求和符号逐 k 累加到 sum_k
    cudaMemset(d_sumk, 0, size * sizeof(float));
    for (int k = 0; k < 2; k++)
    {
        float prod_tau = 1.;
        for (int tau = 1; tau <= 2 * k; tau++)
        {
            prod_tau *= (opt_.v_2 - tau + 1);
        }

        // 计算在当前 k 值的情况下，八个方向的偏微分的差分的和
        //  D_l_power = np.power(D_l, (v_2 - 2*k - 2))
        arrayPower<<<blockGrid1, threadBlock1>>>(d_D_l, d_D_l_power, opt_.v_2 - 2 * k - 2, size);
        // D_ls_power = np.power(D_ls, (v_2 - 2*k - 2))
        arrayPower<<<blockGrid1, threadBlock1>>>(d_D_ls, d_D_ls_power, opt_.v_2 - 2 * k - 2, size);

        // second_term = alpha_1 * np.power((np.abs(ls)),(v_2 - 2*k - 2)) * ls
        dim3 thread{32, 32};
        dim3 block{(column + thread.x - 1) / thread.x,
                   (row + thread.y - 1) / thread.y};
        secondTerm<<<block, thread>>>(d_ls, d_second_term, opt_.v_2 - 2 * k - 2,
                                      opt_.alpha_1, row, column, n);

        // iterate through eight directions
        cudaMemset(d_eight_terms, 0, size * sizeof(float));
        for (int layer = 0; layer < 8; layer++)
        {
            computeTemp<<<block, thread>>>(d_D_l_power, d_D_l_8 + size * layer, d_D_ls_power,
                                           d_D_ls_8 + size * layer, d_second_term,
                                           opt_.alpha_2, row, column, n, d_temp_pad);

            // symetricPadDevice(d_temp_pad, row, column, n);

            cudaMemset(d_temp_results, 0, size * sizeof(float));
            fstDiffCuda(d_temp_pad, d_temp_results, d_mask, layer, row, column);
            // utTestPrintVector(tempResults, "tempResults");

            // utTestPrintDeviceVector(d_temp_results, "d_temp_results", size, size, column);

            // eight_terms = eight_terms + fstDiff(temp, layer)
            addEightTerms<<<blockGrid1, threadBlock1>>>(d_eight_terms, d_temp_results, size);
        }

        computeSumk<<<blockGrid1, threadBlock1>>>(d_sumk, d_eight_terms, prod_tau / tgammaf(2 * k + 1), size);
    }
    // utTestPrintVector(sumk, "sumk");

    // utTestPrintDeviceVector(d_sumk, "d_sumk", size, size, column);

    // do fractional derivative
    computeResults<<<blockGrid1, threadBlock1>>>(d_results, d_sumk,
                                                 -tgammaf(1 - opt_.v_1) / tgammaf(-opt_.v_1) / tgammaf(-opt_.v_3),
                                                 size);
}

// use two dimensional grid and two dimensional block
__global__ void computeDealtaL(float *Delta_l, float *p_l, float *l_curr, const int row, const int column,
                               const int n, const float v3, const float Delta_t, const float mu)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c > column || r > row)
    {
        return;
    }

    int idx1 = r * column + c;
    int idx2 = (r + n) * (column + 2 * n) + c + n;

    if (v3 > 0)
    {
        Delta_l[idx1] = p_l[idx1] * pow(Delta_t, v3) -
                        2 * mu / tgammaf(3 - v3) * (Delta_l[idx1] * Delta_l[idx1]) * 1 / (pow(l_curr[idx2], v3));
    }
    else
    {
        Delta_l[idx1] = p_l[idx1] * pow(Delta_t, v3) -
                        2 * mu / tgammaf(3 - v3) * (Delta_l[idx1] * Delta_l[idx1]) * pow(l_curr[idx2], -v3);
    }
}

void franctionalRetinexCuda(float *l_curr, const float *s, const float *c, const int row, const int column)
{
    float *d_l_curr, *d_s, *d_c, *d_ls;
    float *d_p_l, *d_delta_l;
    float *d_D_l, *d_D_l_s;
    // used in computeP for to store temp results
    float *d_D_l_8, *d_D_ls_8, *d_D_l_power, *d_D_ls_power, *d_eight_terms, *d_second_term,
        *d_temp_pad, *d_temp_results, *d_sumk;

    int size = row * column;
    int paddedSize = (row + 2 * (opt_.N - 2)) * (column + 2 * (opt_.N - 2));
    int imgSize = sizeof(float) * size;
    int paddedImgSize = sizeof(float) * paddedSize;
    int templateSize = sizeof(float) * opt_.N;

    // prepare memory on device
    printIfCudaFailed(cudaMalloc(&d_l_curr, paddedImgSize));
    printIfCudaFailed(cudaMalloc(&d_s, paddedImgSize));
    printIfCudaFailed(cudaMalloc(&d_ls, paddedImgSize));
    printIfCudaFailed(cudaMalloc(&d_c, templateSize));
    printIfCudaFailed(cudaMalloc(&d_p_l, imgSize));
    printIfCudaFailed(cudaMalloc(&d_delta_l, imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l, imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l_s, imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l_8, 8 * imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_ls_8, 8 * imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l_power, imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_ls_power, imgSize));
    printIfCudaFailed(cudaMalloc(&d_eight_terms, imgSize));
    printIfCudaFailed(cudaMalloc(&d_second_term, imgSize));
    printIfCudaFailed(cudaMalloc(&d_temp_pad, paddedImgSize));
    printIfCudaFailed(cudaMalloc(&d_temp_results, imgSize));
    printIfCudaFailed(cudaMalloc(&d_sumk, imgSize));

    printIfCudaFailed(cudaMemcpy(d_l_curr, l_curr, paddedImgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(d_s, s, paddedImgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(d_c, c, templateSize, cudaMemcpyHostToDevice));

    int n = opt_.N - 2;
    dim3 threadBlock{32, 32};
    dim3 blockGrid{(column + threadBlock.x - 1) / threadBlock.x, (row + threadBlock.y - 1) / threadBlock.y};
    dim3 Block{64, 1};
    dim3 Grid{(paddedSize + Block.x - 1) / Block.x, 1};

    for (int t = 0; t < opt_.n; t++)
    {
        computeP(d_l_curr, d_s, d_c, d_ls, d_p_l, d_D_l, d_D_l_s, d_D_l_8, d_D_ls_8,
                 d_D_l_power, d_D_ls_power, d_eight_terms, d_second_term,
                 d_temp_pad, d_temp_results, d_sumk, row, column);

        // utTestPrintDeviceVector(d_p_l, "p_l", size, size, column);

        // debug
        cout << "computeP " << endl;

        computeDealtaL<<<blockGrid, threadBlock>>>(d_delta_l, d_p_l, d_l_curr, row, column, n,
                                                   opt_.v_3, opt_.Delta_t, opt_.mu);

        //  l_curr = l_curr + Delta_l
        update_l_curr<<<blockGrid, threadBlock>>>(d_l_curr, d_delta_l, row, column, n);

        // l_curr = np.where(l_curr <= 0, epsilon_2, l_curr)
        imageRelpaceWith<<<Grid, Block>>>(d_l_curr, opt_.epsilon_2, paddedSize);
    }

    // L = np.exp(l_curr)
    imageExp<<<Grid, Block>>>(d_l_curr, paddedSize);

    printIfCudaFailed(cudaMemcpy(l_curr, d_l_curr, paddedImgSize, cudaMemcpyDeviceToHost));

    // free memory
    printIfCudaFailed(cudaFree(d_l_curr));
    printIfCudaFailed(cudaFree(d_s));
    printIfCudaFailed(cudaFree(d_c));
    printIfCudaFailed(cudaFree(d_ls));
    printIfCudaFailed(cudaFree(d_p_l));
    printIfCudaFailed(cudaFree(d_delta_l));
    printIfCudaFailed(cudaFree(d_D_l));
    printIfCudaFailed(cudaFree(d_D_l_s));
    printIfCudaFailed(cudaFree(d_D_l_8));
    printIfCudaFailed(cudaFree(d_D_ls_8));
    printIfCudaFailed(cudaFree(d_D_l_power));
    printIfCudaFailed(cudaFree(d_D_ls_power));
    printIfCudaFailed(cudaFree(d_eight_terms));
    printIfCudaFailed(cudaFree(d_second_term));
    printIfCudaFailed(cudaFree(d_temp_pad));
    printIfCudaFailed(cudaFree(d_temp_results));
    printIfCudaFailed(cudaFree(d_sumk));
}

void utTestPadCuda(float *img, const int row, const int column, const int n)
{
    float *d_img;
    int paddedImgSize = sizeof(float) * (row + 2 * n) * (column + 2 * n);

    printIfCudaFailed(cudaMalloc(&d_img, paddedImgSize));
    printIfCudaFailed(cudaMemcpy(d_img, img, paddedImgSize, cudaMemcpyHostToDevice));

    symetricPadDevice(d_img, row, column, n);

    printIfCudaFailed(cudaMemcpy(img, d_img, paddedImgSize, cudaMemcpyDeviceToHost));
    printIfCudaFailed(cudaFree(d_img));
}

void utTestCal8DrivCuda(float *img, float *D_l, float *D_l_8, const float *mask,
                        const int row, const int column, const int n)
{
    int size = row * column;
    int paddedSize = (row + 2 * n) * (column + 2 * n);
    int imgSize = sizeof(float) * size;
    int paddedImgSize = sizeof(float) * paddedSize;

    float *d_img, *d_D_l, *d_D_l_8, *d_mask;
    printIfCudaFailed(cudaMalloc(&d_img, paddedImgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l, imgSize));
    printIfCudaFailed(cudaMalloc(&d_D_l_8, 8 * imgSize));
    printIfCudaFailed(cudaMalloc(&d_mask, sizeof(float) * (n + 2)));

    printIfCudaFailed(cudaMemcpy(d_img, img, paddedImgSize, cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemcpy(d_mask, mask, sizeof(float) * (n + 2), cudaMemcpyHostToDevice));
    printIfCudaFailed(cudaMemset(d_D_l, 0, imgSize));

    symetricPadDevice(d_img, row, column, n);
    cal8DerivCuda(d_img, d_D_l, d_D_l_8, d_mask, row, column);

    printIfCudaFailed(cudaMemcpy(D_l, d_D_l, imgSize, cudaMemcpyDeviceToHost));
    printIfCudaFailed(cudaMemcpy(D_l_8, d_D_l_8, 8 * imgSize, cudaMemcpyDeviceToHost));

    printIfCudaFailed(cudaFree(d_img));
    printIfCudaFailed(cudaFree(d_D_l));
    printIfCudaFailed(cudaFree(d_D_l_8));
    printIfCudaFailed(cudaFree(d_mask));
}