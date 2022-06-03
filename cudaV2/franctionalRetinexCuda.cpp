/*
This version optimize the former one in global memory transfer
*/
#include "./kernels/kernels.h"
#include "utills.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

opt opt_{1.25, 2.25,  0.90,    0.1, 0.05, 0.1, 0.002,
         6,    0.006, 0.00001, 2.2, 7,    2,   false};

using namespace std;

void PU2Operator(vector<float> &mask) {
  // PU2OPERATOR PU-2 分数阶微分算子掩模
  //   给定阶数 v 和掩模尺寸 N = n + 2，其中 N >= 3，N 通常为奇数
  //   返回掩模系数，从 C_{-1} 到 C_n

  int n = opt_.N - 2;
  float v = opt_.v_1;
  vector<float> temp{v / 4 + v * v / 8, 1 - v * v / 4, -v / 4 + v * v / 8};

  for (int k = -1; k < n - 1; k++) {
    float _k = k + opt_.epsilon_2;
    mask[k + 1] = 1 / tgammaf(-v) *
                  (tgammaf(_k - v + 1) / tgammaf(_k + 2) * temp[0] +
                   tgammaf(_k - v) / tgammaf(_k + 1) * temp[1] +
                   tgammaf(_k - v - 1) / tgammaf(_k) * temp[2]);
  }

  mask[n] = tgammaf(n - v - 1) / tgammaf(n) / tgammaf(-v) * temp[1] +
            tgammaf(n - v - 2) / tgammaf(n - 1) / tgammaf(-v) * temp[2];
  mask[n + 1] = tgammaf(n - v - 1) / tgammaf(n) / tgammaf(-v) * temp[2];
}

inline void copyImgsBack(unsigned char *img, float *vChannelImg, float *l_curr,
                         int row, int column, int n) {
  int paddedColumn = column + 2 * n;
#pragma omp parallel for
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      int idx1 = i * column + j;
      int idx2 = (i + n) * paddedColumn + j + n;
      img[idx1] = (unsigned char)(255 * (vChannelImg[idx1] /
                                         pow(l_curr[idx2] / 255.0,
                                             1 - 1 / opt_.gamma_cor)));
    }
  }
}

inline void vchannelImgNormalize(float *img, float v_max, float v_min,
                                 int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    img[i] = (img[i] - v_min) / (v_max - v_min);
  }
}

inline void computeS(float *s, float *img, int row, int column, int n) {
  // s
  int paddedColumn = column + 2 * n;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      int idx1 = i * column + j;
      int idx2 = (i + n) * paddedColumn + j + n;
      s[idx2] = log(255 * img[idx1] + 1);
    }
  }
}

inline void computeL_curr(float *l_curr, float *s, int row, int column, int n) {
  int paddedColumn = column + 2 * n;
#pragma omp parallel for
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      int idx2 = (i + n) * paddedColumn + j + n;

      l_curr[idx2] = 1.05 * s[idx2];
      if (l_curr[idx2] < opt_.epsilon_2) {
        l_curr[idx2] = opt_.epsilon_2;
      }
    }
  }
}

inline void copyVchannelImg(float *vchannelImg, unsigned char *channels,
                            int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    vchannelImg[i] = channels[i];
  }
}

inline void findMaxMin(const vector<float> &v, float *max, float *min) {
  *min = 10000, *max = -1;
  for (auto iter : v) {
    if (iter > *max) {
      *max = iter;
    }
    if (iter < *min) {
      *min = iter;
    }
  }
}

void franctionalRetinex(cv::Mat &bgrImg, int row, int column) {
  void utTestPrintVec(vector<float> & v, int column);

  /* get vchannel image */
  cv::Mat hsvImg;
  cv::cvtColor(bgrImg, hsvImg, cv::COLOR_RGB2HSV);
  vector<cv::Mat> channels;
  cv::split(hsvImg, channels);

  // debug
  if (opt_.debug) {
    cv::imwrite("testh.jpg", channels[0]);
    cv::imwrite("tests.jpg", channels[1]);
    cv::imwrite("testv.jpg", channels[2]);
  }

  // debug
  // row = 6;
  // column = 6;

  int imgSize = row * column;
  int n = opt_.N - 2;
  int paddedImgSize = (row + 2 * n) * (column + 2 * n);

  /* prepare data */
  vector<float> vchannelImg(imgSize, 0.);

  // debug
  copyVchannelImg(&(vchannelImg[0]), channels[2].data, row * column);
  // for (int i = 0; i < row * column; i++) {
  //  vchannelImg[i] = i;
  // }

  float v_min = 10000, v_max = -1;
  findMaxMin(vchannelImg, &v_max, &v_min);

  // debug
  cout << "vmax: " << v_max << " vmin " << v_min << endl;
  // use bigger space to avoid redundant memory aecces in pad operation
  vector<float> s(paddedImgSize, 0.);
  vector<float> l_curr(paddedImgSize, 0.);

  omp_set_num_threads(4);
  vchannelImgNormalize(&(vchannelImg[0]), v_max, v_min, row * column);

  computeS(&(s[0]), &(vchannelImg[0]), row, column, n);

  computeL_curr(&(l_curr[0]), &(s[0]), row, column, n);

  vector<float> C{0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020};
  PU2Operator(C);

  // this function will change l_curr
  franctionalRetinexCuda(&(l_curr[0]), &(s[0]), &(C[0]), row, column);

  // copy images back
  copyImgsBack(channels[2].data, &(vchannelImg[0]), &(l_curr[0]), row, column,
               n);

  // vChannel = (255 * vChannel/(np.power((L/255), (1 -
  // 1/gamma_corr)))).astype(np.uint8)

  // debug

  cv::merge(channels, hsvImg);

  cv::Mat processedImg;
  cv::cvtColor(hsvImg, processedImg, cv::COLOR_HSV2RGB);
  cv::imwrite("./processImg.jpg", processedImg);
}

void utTestPrintVec(vector<float> &v, int column) {
  for (int i = 0; i < v.size(); i++) {
    cout << v[i] << "\t";
    if ((i + 1) % column == 0) {
      cout << "\n";
      continue;
    }
  }
  cout << "\n" << endl;
}

void utTestPad() {
  int row{6}, column{6}, n{5};
  vector<float> a(row * column, 0);
  for (int i = 0; i < row * column; i++) {
    a[i] = i;
  }

  vector<float> aPad((row + 2 * n) * (column + 2 * n), 0);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      aPad[(i + n) * (column + 2 * n) + j + n] = a[i * column + j];
    }
  }

  utTestPrintVec(aPad, column + 2 * n);
  utTestPadCuda(&(aPad[0]), row, column, n);
  utTestPrintVec(aPad, column + 2 * n);
}

void utTestPrintD_l_8(vector<float> &D_l_8, const int size, const int layer) {
  for (int i = 0; i < size; i++) {
    cout << D_l_8[size * layer + i] << " ";
  }

  cout << "\n" << endl;
}

void utTestCal8Deriv() {
  int row{6}, column{6}, n{5};
  int size{row * column};
  int paddedSize{(row + 2 * n) * (column + 2 * n)};

  vector<float> a(row * column, 0);
  for (int i = 0; i < row * column; i++) {
    a[i] = i;
  }

  vector<float> aPad((row + 2 * n) * (column + 2 * n), 0);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++) {
      aPad[(i + n) * (column + 2 * n) + j + n] = a[i * column + j];
    }
  }

  // utTestPrintVec(aPad);

  vector<float> D_l(size, 0);
  vector<float> D_l_8(8 * size, 0);
  vector<float> mask{0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020};

  PU2Operator(mask);

  // utTestPrintVec(mask);

  utTestCal8DrivCuda(&(aPad[0]), &(D_l[0]), &(D_l_8[0]), &(mask[0]), row,
                     column, n);

  utTestPrintVec(D_l, column);

  utTestPrintD_l_8(D_l_8, size, 0);
  utTestPrintD_l_8(D_l_8, size, 1);
  utTestPrintD_l_8(D_l_8, size, 2);
  utTestPrintD_l_8(D_l_8, size, 3);
  utTestPrintD_l_8(D_l_8, size, 4);
  utTestPrintD_l_8(D_l_8, size, 5);
  utTestPrintD_l_8(D_l_8, size, 6);
  utTestPrintD_l_8(D_l_8, size, 7);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "usage: ./franctionalRetinex imgPath" << endl;
    return -1;
  }

  const char *imagepath = argv[1];
  cv::Mat m = cv::imread(imagepath, 1);
  if (m.empty()) {
    cout << "cv::imread " << imagepath << " failed" << endl;
    return -1;
  }

  // utTestPad();
  // utTestCal8Deriv();

  auto start = std::chrono::steady_clock::now();
  franctionalRetinex(m, m.rows, m.cols);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> elapsed =
      end - start; // std::micro 表示以微秒为时间单位
  std::cout << "time: " << elapsed.count() << "us" << std::endl;

  return 0;
}