#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define printIfCudaFailed(expression)                                      \
    if ((expression) != cudaSuccess)                                       \
                                                                           \
    {                                                                      \
        std::cout << "cuda failed: " << __FILE__ << __LINE__ << std::endl; \
    }

extern texture<float, 2> texRef;

extern __global__ void directionTop(const float *img, const float *weights,
                                    const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionBottom(const float *img, const float *weights,
                                       const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionLeft(const float *img, const float *weights,
                                     const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionRight(const float *img, const float *weights,
                                      const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionLeftBottom(const float *img,
                                           const float *weights, const int n,
                                           const int group, const int column, const int row, float *result);

extern __global__ void directionRightTop(const float *img, const float *weights,
                                         const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionLeftTop(const float *img, const float *weights,
                                        const int n, const int group, const int column, const int row, float *result);

extern __global__ void directionRightBottom(const float *img,
                                            const float *weights, const int n,
                                            const int group, const int column, const int row, float *result);

extern void dTopRemain(const int N, const int row, const int column,
                       const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dBottomRemain(const int N, const int row, const int column,
                          const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dLeftRemain(const int N, const int row, const int column,
                        const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dRightRemain(const int N, const int row, const int column,
                         const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dLeftBottomRemain(const int N, const int row, const int column,
                              const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dRightTopRemain(const int N, const int row, const int column,
                            const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dLeftTopRemain(const int N, const int row, const int column,
                           const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern void dRightBottomRemain(const int N, const int row, const int column,
                               const int rowRemain, const int colRemain, const float *img, const float *weights, float *result);

extern __global__ void powerKernel(const float *img, float *result, float power);

extern void powerRemain(const float *img, float *result, float power, int remain, int size);

extern __global__ void secondTermKernel(const float *img, float *result, float alpha, float power);

extern void secondTermRemain(const float *img, float *result, float alpha, float power, int remain, int size);

extern __global__ void tempKernel(const float *D_l_power, const float *D_l_8, const float *D_ls_8, const float *second_term,
                                  const float *D_ls_power, float alpha, float *temp);

extern void tempRemain(const float *D_l_power, const float *D_l_8, const float *D_ls_8, const float *second_term,
                       const float *D_ls_power, float alpha, float *temp, int size, int remain);