#include <iostream>

typedef struct opt {
  float v_1;
  float v_2;
  float v_3;
  float mu;
  float alpha_1;
  float alpha_2;
  float Delta_t;
  int n;
  float epsilon_1;
  float epsilon_2;
  float gamma_cor;
  int N;
  int norm;
  bool debug;
} opt;
extern opt opt_;

#define printIfCudaFailed(expression)                                          \
  if ((expression) != cudaSuccess)                                             \
                                                                               \
  {                                                                            \
    std::cout << "cuda failed: " << __FILE__ << __LINE__ << std::endl;         \
  }
