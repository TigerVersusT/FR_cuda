#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

struct opt {
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
} opt{1.25, 2.25,  0.90,    0.1, 0.05, 0.1, 0.002,
      6,    0.006, 0.00001, 2.2, 7,    2,   false};

void PU2Operator(vector<float> &mask) {
  // PU2OPERATOR PU-2 分数阶微分算子掩模
  //   给定阶数 v 和掩模尺寸 N = n + 2，其中 N >= 3，N 通常为奇数
  //   返回掩模系数，从 C_{-1} 到 C_n

  int n = opt.N - 2;
  float v = opt.v_1;
  vector<float> temp{v / 4 + v * v / 8, 1 - v * v / 4, -v / 4 + v * v / 8};

  for (int k = -1; k < n - 1; k++) {
    float _k = k + opt.epsilon_2;
    mask[k + 1] = 1 / tgammaf(-v) *
                  (tgammaf(_k - v + 1) / tgammaf(_k + 2) * temp[0] +
                   tgammaf(_k - v) / tgammaf(_k + 1) * temp[1] +
                   tgammaf(_k - v - 1) / tgammaf(_k) * temp[2]);
  }

  mask[n] = tgammaf(n - v - 1) / tgammaf(n) / tgammaf(-v) * temp[1] +
            tgammaf(n - v - 2) / tgammaf(n - 1) / tgammaf(-v) * temp[2];
  mask[n + 1] = tgammaf(n - v - 1) / tgammaf(n) / tgammaf(-v) * temp[2];
}

void iterativePad(const vector<float> &img, vector<float> &paddedImg,
                  const int n, const int row, const int column) {

  // expand border recursively
  // copy before padding
  int pRow = row + n * 2;
  int pColoumn = column + n * 2;

  // cout << "n: " << n << " row: " << row << " column: " << column << "
  // pColumn: " << pColoumn << endl;

  for (int r = n; r < row + n; r++) {
    for (int c = n; c < column + n; c++) {
      paddedImg[r * pColoumn + c] = img[(r - n) * column + c - n];
    }
  }

  for (int i = 0; i < n; i++) {
    // pad horizontally
    int startRow = n - i, endRow = startRow + row + 2 * i;
    int leftCol = n - i - 1, rightCol = n + column + i;

    for (int r = startRow; r < endRow; r++) {
      // pad left

      paddedImg[r * pColoumn + leftCol] =
          3 * (paddedImg[r * pColoumn + leftCol + 1] -
               paddedImg[r * pColoumn + leftCol + 2]) +
          paddedImg[r * pColoumn + leftCol + 3];

      // pad right
      paddedImg[r * pColoumn + rightCol] =
          3 * (paddedImg[r * pColoumn + rightCol - 1] -
               paddedImg[r * pColoumn + rightCol - 2]) +
          paddedImg[r * pColoumn + rightCol - 3];
    }

    // pad vertically
    int startCol = n - i, endCol = startCol + column + 2 * i;
    int topRow = n - 1 - i, bottomRow = n + row + i;
    for (int c = startCol; c < endCol; c++) {

      // pad top
      paddedImg[topRow * pColoumn + c] =
          3 * (paddedImg[(topRow + 1) * pColoumn + c] -
               paddedImg[(topRow + 2) * pColoumn + c]) +
          paddedImg[(topRow + 3) * pColoumn + c];

      // pad bottom
      paddedImg[bottomRow * pColoumn + c] =
          3 * (paddedImg[(bottomRow - 1) * pColoumn + c] -
               paddedImg[(bottomRow - 2) * pColoumn + c]) +
          paddedImg[(bottomRow - 3) * pColoumn + c];
    }

    paddedImg[(startRow - 1) * pColoumn + startCol - 1] =
        3 * (paddedImg[(startRow)*pColoumn + startCol] -
             paddedImg[(startRow + 1) * pColoumn + startCol + 1]) +
        paddedImg[(startRow + 2) * pColoumn + startCol + 2];
    // cout << "padded left top! " << endl;

    paddedImg[(startRow - 1) * pColoumn + endCol] =
        3 * (paddedImg[(startRow)*pColoumn + endCol - 1] -
             paddedImg[(startRow + 1) * pColoumn + endCol - 2]) +
        paddedImg[(startRow + 2) * pColoumn + endCol - 3];
    // cout << "padded right top! " << endl;

    paddedImg[(endRow)*pColoumn + startCol - 1] =
        3 * (paddedImg[(endRow - 1) * pColoumn + startCol] -
             paddedImg[(endRow - 2) * pColoumn + startCol + 1]) +
        paddedImg[(endRow - 3) * pColoumn + startCol + 2];
    // cout << "padded left bottom! " << endl;

    paddedImg[(endRow)*pColoumn + endCol] =
        3 * (paddedImg[(endRow - 1) * pColoumn + endCol - 1] -
             paddedImg[(endRow - 2) * pColoumn + endCol - 2]) +
        paddedImg[(endRow - 3) * pColoumn + endCol - 3];
    // cout << "padded right bottom " << endl;
  }
}

void pad(const vector<float> &img, vector<float> &paddedImg, const int n,
         const int row, const int column) {
  // PAD 对图像矩阵的边缘进行拉格朗日插值延拓
  // img 为图像矩阵，n 为延拓次数，拉格朗日插值公式为 s(-1) = 3[s(0) - s(1)] +
  // s(2) row, column 为 A 的行数和列数

  // copy before padding
  int pRow = row + n * 2;
  int pColoumn = column + n * 2;

  // cout << "n: " << n << " row: " << row << " column: " << column << "
  // pColumn: " << pColoumn << endl;

  for (int r = n; r < row + n; r++) {
    for (int c = n; c < column + n; c++) {
      paddedImg[r * pColoumn + c] = img[(r - n) * column + c - n];
    }
  }

  // padding
  // pad horizontally
  for (int r = n; r < n + row; r++) {
    // pad left
    for (int c = n - 1; c >= 0; c--) {
      paddedImg[r * pColoumn + c] = 3 * (paddedImg[r * pColoumn + c + 1] -
                                         paddedImg[r * pColoumn + c + 2]) +
                                    paddedImg[r * pColoumn + c + 3];
    }
    // pad right
    for (int c = n + column; c < pColoumn; c++) {
      paddedImg[r * pColoumn + c] = 3 * (paddedImg[r * pColoumn + c - 1] -
                                         paddedImg[r * pColoumn + c - 2]) +
                                    paddedImg[r * pColoumn + c - 3];
    }
  }

  // cout << "padded horizontally! " << endl;

  // pad vertically
  for (int c = n; c < n + column; c++) {
    // pad top
    for (int r = n - 1; r >= 0; r--) {
      paddedImg[r * pColoumn + c] = 3 * (paddedImg[(r + 1) * pColoumn + c] -
                                         paddedImg[(r + 2) * pColoumn + c]) +
                                    paddedImg[(r + 3) * pColoumn + c];
    }
    // pad bottom
    for (int r = n + row; r < pRow; r++) {
      paddedImg[r * pColoumn + c] = 3 * (paddedImg[(r - 1) * pColoumn + c] -
                                         paddedImg[(r - 2) * pColoumn + c]) +
                                    paddedImg[(r - 3) * pColoumn + c];
    }
  }

  // cout << "padded vertically! " << endl;

  // pad four corners
  for (int r = n - 1; r >= 0; r--) {
    for (int c = n - 1; c >= 0; c--) {
      paddedImg[r * pColoumn + c] =
          3 * (paddedImg[(r + 1) * pColoumn + c + 1] -
               paddedImg[(r + 2) * pColoumn + c + 2]) +
          paddedImg[(r + 3) * pColoumn + c + 3];
    }
  }
  // cout << "padded left top! " << endl;

  for (int r = n - 1; r >= 0; r--) {
    for (int c = n + column; c < pColoumn; c++) {
      paddedImg[r * pColoumn + c] =
          3 * (paddedImg[(r + 1) * pColoumn + c - 1] -
               paddedImg[(r + 2) * pColoumn + c - 2]) +
          paddedImg[(r + 3) * pColoumn + c - 3];
    }
  }
  // cout << "padded right top! " << endl;

  for (int r = row + n; r < pRow; r++) {
    for (int c = n - 1; c >= 0; c--) {
      paddedImg[r * pColoumn + c] =
          3 * (paddedImg[(r - 1) * pColoumn + c + 1] -
               paddedImg[(r - 2) * pColoumn + c + 2]) +
          paddedImg[(r - 3) * pColoumn + c + 3];
    }
  }
  // cout << "padded left bottom! " << endl;

  for (int r = row + n; r < pRow; r++) {
    for (int c = column + n; c < pColoumn; c++) {
      paddedImg[r * pColoumn + c] =
          3 * (paddedImg[(r - 1) * pColoumn + c - 1] -
               paddedImg[(r - 2) * pColoumn + c - 2]) +
          paddedImg[(r - 3) * pColoumn + c - 3];
    }
  }
  // cout << "padded right bottom " << endl;
}

void cal8Deriv(const vector<float> &vChanelImg, vector<vector<float>> &D8,
               vector<float> &D, const vector<float> &mask, const int row,
               const int column, const int norm = 1) {
  // CAL8DERIV 计算八个方向上的分数阶偏导数
  // 根据系数向量，对输入进来的矩阵计算八个方向上的偏导数矩阵
  // C 的长度为 n + 2
  // D8 的第三个维度的排列顺序为 u, d, l, r, ld, ru, lu, rd
  int n = opt.N - 2;

  int pRow = row + n * 2;
  int pColumn = column + n * 2;
  vector<float> paddedImg(pRow * pColumn, 0.);

  iterativePad(vChanelImg, paddedImg, n, row, column);

  /*
  vector<float> paddedImg{
 -19,   -18,   -17,   -16,   -15,   -14,   -13,   -12,   -11,   -11,   -11, -11,
 -11, -16,   -15,   -14,   -13,   -12,   -11,   -10,    -9,    -8,    -8,    -8,
 -8,    -8, -13,   -12,   -11,   -10,    -9,    -8,    -7,    -6,    -5,    -5,
 -5,    -5,    -5, -10,    -9,    -8,    -7,    -6,    -5,    -4,    -3,    -2,
 -2,    -2,    -2,    -2, -7,    -6,    -5,    -4,    -3,    -2,    -1,     0,
 1,     1,     1,     1,     1, -4,    -3,    -2,    -1,     0,     1,     2, 3,
 4,     4,     4,     4,     4, -1,     0,     1,     2,     3,     4,     5, 6,
 7,     7,     7,     7,     7, 2,     3,     4,     5,     6,     7,     8, 9,
 10,    10,    10,    10,    10, 5,     6,     7,     8,     9,    10,    11,
 12,    13,    13,    13,    13,    13, 5,     6,     7,     8,     9,    10,
 11,    12,    13,    13,    13,    13,    13, 5,     6,     7,     8,     9,
 10,    11,    12,    13,    13,    13,    13,    13, 5,     6,     7,     8, 9,
 10,    11 ,   12,    13,    13,    13,    13,    13, 5,     6,     7,     8, 9,
 10,    11,    12,    13,    13,    13,    13,    13};
  */
  // 分别计算八个方向的分数阶偏导数矩阵
  vector<float> &dTop = D8[0];

  for (int k = 0; k < opt.N; k++) {
    int startRow = k;
    int startCol = n;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[opt.N - k - 1];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dTop[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dBottom = D8[1];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n - 1 + k;
    int startCol = n;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dBottom[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dLeft = D8[2];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n;
    int startCol = k;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[opt.N - k - 1];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dLeft[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dRight = D8[3];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n;
    int startCol = n - 1 + k;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dRight[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dLeftBottom = D8[4];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n + k - 1;
    int startCol = n - k + 1;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dLeftBottom[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dRightTop = D8[5];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n + 1 - k;
    int startCol = n - 1 + k;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dRightTop[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dLeftTop = D8[6];
  for (int k = 0; k < opt.N; k++) {
    int startRow = n + 1 - k;
    int startCol = n + 1 - k;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dLeftTop[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  vector<float> &dRightBottom = D8[7];

  for (int k = 0; k < opt.N; k++) {
    int startRow = n - 1 + k;
    int startCol = n - 1 + k;
    int endRow = startRow + row;
    int endCol = startCol + column;

    auto weight = mask[k];
    for (int r = startRow; r < endRow; r++) {
      for (int c = startCol; c < endCol; c++) {
        dRightBottom[(r - startRow) * column + c - startCol] +=
            weight * paddedImg[r * pColumn + c];
      }
    }
  }

  // 计算全微分
  if (norm == 1) {
    for (int i = 0; i < row * column; i++) {
      D[i] += abs(dTop[i]) + abs(dBottom[i]) + abs(dLeft[i]) + abs(dRight[i]) +
              abs(dLeftBottom[i]) + abs(dRightTop[i]) + abs(dLeftTop[i]) +
              abs(dRightBottom[i]);
    }
  }
}

void fstDiff(const vector<float> &A, vector<float> &result, const int direction,
             const int row, const int column) {
  // FSTDIFF 对一个矩阵在指定方向上做一次差分,采用一阶分数阶微分方式
  //   对输入矩阵 A, 向 direction 方向做一次差分，采用一节分数阶微分方式
  // direction 设置如下:
  // [0, 1, 2, 3, 4, 5, 6, 7] := [D_u, D_d, D_l, D_r, D_ld, D_ru, D_lu, D_rd]

  void utTestPrintVector(const vector<float> &vec, const string prefix);

  vector<float> mask{0.375, 0.375, -0.875, 0.125};
  int n = 2;

  int pRow = row + n * 2;
  int pColumn = column + n * 2;
  vector<float> A_pad(pRow * pColumn, 0.);
  iterativePad(A, A_pad, n, row, column);

  for (int i = 0; i < row * column; i++) {
    result[i] = 0.;
  }

  /*
    vector<float> A_pad{-7, -6, -5, -4, -3, -2, -2, -4, -3, -2, -1, 0,  1,
                        1,  -1, 0,  1,  2,  3,  4,  4,  2,  3,  4,  5,  6,
                        7,  7,  5,  6,  7,  8,  9,  10, 10, 8,  9,  10, 11,
                        12, 13, 13, 8,  9,  10, 11, 12, 13, 13};
  */
  if (direction == 0) // D_u
  {
    for (int k = 0; k < 4; k++) {
      int startRow = k;
      int startCol = n;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[4 - k - 1];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 1) // D_d
  {
    for (int k = 0; k < 4; k++) {
      int startRow = 1 + k;
      int startCol = n;
      int endRow = k + row + 1;
      int endCol = 2 + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 2) // D_l
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n;
      int startCol = k;
      int endRow = n + row;
      int endCol = k + column;

      auto weight = mask[4 - k - 1];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 3) // D_r
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n;
      int startCol = n - 1 + k;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 4) // D_ld
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n + k - 1;
      int startCol = n - k + 1;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 5) // D_ru
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n + 1 - k;
      int startCol = n - 1 + k;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else if (direction == 6) // D_lu
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n + 1 - k;
      int startCol = n + 1 - k;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  } else // D_rd
  {
    for (int k = 0; k < 4; k++) {
      int startRow = n - 1 + k;
      int startCol = n - 1 + k;
      int endRow = startRow + row;
      int endCol = startCol + column;

      auto weight = mask[k];
      for (int r = startRow; r < endRow; r++) {
        for (int c = startCol; c < endCol; c++) {
          result[(r - startRow) * column + c - startCol] +=
              weight * A_pad[r * pColumn + c];
        }
      }
    }
  }
}

void computeP(const vector<float> &l, vector<float> &s_0,
              const vector<float> &C, vector<float> &results, const int row,
              const int column, const int norm) {
  // computeP 根据当前矩阵 l^n，利用给定参数 α_1, α_2，计算 P(l^n)
  // 针对当前的 l 矩阵，分别计算 l 和 (l - s) 在八个方向上的分数阶偏导数矩阵
  void utTestPrintVector(const vector<float> &vec, const string prefix);

  vector<float> ls(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    ls[i] = l[i] - s_0[i];
    if (ls[i] <= opt.epsilon_2) {
      ls[i] = opt.epsilon_2;
    }
  }

  /* calculate D8_l and D_l */
  /* allocate space to save D8 and D*/
  vector<vector<float>> D_l_8;
  for (int i = 0; i < 8; i++) {
    vector<float> temp(row * column, 0.);
    D_l_8.push_back(temp);
  }
  vector<float> D_l(row * column, 0.);
  cal8Deriv(l, D_l_8, D_l, C, row, column, norm);
  for (int i = 0; i < row * column; i++) {
    if (D_l[i] < opt.epsilon_1) {
      D_l[i] = opt.epsilon_1;
    }
  }
  utTestPrintVector(D_l, "D_l");

  /* calculate D8_ls and D_ls */
  vector<vector<float>> D_ls_8;
  for (int i = 0; i < 8; i++) {
    vector<float> temp(row * column, 0.);
    D_ls_8.push_back(temp);
  }
  vector<float> D_ls(row * column, 0.);
  cal8Deriv(ls, D_ls_8, D_ls, C, row, column, norm);
  for (int i = 0; i < row * column; i++) {
    if (D_ls[i] < opt.epsilon_1) {
      D_ls[i] = opt.epsilon_1;
    }
  }
  utTestPrintVector(D_ls, "D_ls");

  // 对 k 进行遍历，将求和符号逐 k 累加到 sum_k
  vector<float> sumk(row * column, 0.);
  for (int k = 0; k < 2; k++) {

    float prod_tau = 1.;
    for (int tau = 1; tau <= 2 * k; tau++) {
      prod_tau *= (opt.v_2 - tau + 1);
    }

    // 计算在当前 k 值的情况下，八个方向的偏微分的差分的和
    //  D_l_power = np.power(D_l, (v_2 - 2*k - 2))
    vector<float> D_l_power(row * column, 0.);
    for (int i = 0; i < row * column; i++) {
      if (opt.v_2 - 2 * k - 2 < 0) {
        D_l_power[i] = 1 / (pow(D_l[i], abs(opt.v_2 - 2 * k - 2)));
      } else {
        D_l_power[i] = pow(D_l[i], opt.v_2 - 2 * k - 2);
      }
    }

    // D_ls_power = np.power(D_ls, (v_2 - 2*k - 2))
    vector<float> D_ls_power(row * column, 0.);
    for (int i = 0; i < row * column; i++) {
      if (opt.v_2 - 2 * k - 2 < 0) {
        D_ls_power[i] = 1 / (pow(D_ls[i], abs(opt.v_2 - 2 * k - 2)));
      } else {
        D_ls_power[i] = pow(D_ls[i], opt.v_2 - 2 * k - 2);
      }
    }

    vector<float> eight_terms(row * column, 0.);
    // second_term = alpha_1 * np.power((np.abs(ls)),(v_2 - 2*k - 2)) * ls
    vector<float> second_term(row * column, 0.);
    for (int i = 0; i < row * column; i++) {
      if (opt.v_2 - 2 * k - 2 < 0) {
        second_term[i] =
            (opt.alpha_1 / (pow(abs(ls[i]), abs(opt.v_2 - 2 * k - 2)))) * ls[i];
      } else {
        second_term[i] =
            opt.alpha_1 * pow(abs(ls[i]), opt.v_2 - 2 * k - 2) * ls[i];
      }
    }

    utTestPrintVector(second_term, "second_term");

    vector<float> temp(row * column, 0.);
    vector<float> tempResults(row * column, 0.);
    for (int layer = 0; layer < 8; layer++) {

      // temp = D_l_power * D_l_8[layer] + second_term + alpha_2 * D_ls_power *
      // D_ls_8[layer]
      for (int i = 0; i < row * column; i++) {
        temp[i] = D_l_power[i] * D_l_8[layer][i] + second_term[i] +
                  opt.alpha_2 * D_ls_power[i] * D_ls_8[layer][i];
      }

      utTestPrintVector(temp, "temp[i]");

      fstDiff(temp, tempResults, layer, row, column);

      utTestPrintVector(tempResults, "tempResults");

      // eight_terms = eight_terms + fstDiff(temp, layer)
      for (int i = 0; i < row * column; i++) {
        eight_terms[i] += tempResults[i];
      }
    }

    for (int i = 0; i < row * column; i++) {
      sumk[i] += prod_tau / tgammaf(2 * k + 1) * eight_terms[i];
    }
  }

  utTestPrintVector(sumk, "sumk");

  // do fractional derivative
  for (int i = 0; i < row * column; i++) {
    results[i] =
        -tgammaf(1 - opt.v_1) / tgammaf(-opt.v_1) / tgammaf(-opt.v_3) * sumk[i];
  }

  // P_l = -math.gamma(1 - v_1)/math.gamma(-v_1)/math.gamma(-v_3) * sum_k
}

void franctionalRetinex(cv::Mat &bgrImg, int row, int column) {
  /* get vchannel image */
  cv::Mat hsvImg;
  cv::cvtColor(bgrImg, hsvImg, cv::COLOR_RGB2HSV);
  vector<cv::Mat> channels;
  cv::split(hsvImg, channels);

  // debug
  cv::imwrite("testh.jpg", channels[0]);
  cv::imwrite("tests.jpg", channels[1]);
  cv::imwrite("testv.jpg", channels[2]);

  vector<float> vchannelImg(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    vchannelImg[i] = channels[2].data[i];
  }

  /* prepare data */
  float v_min = 10000, v_max = -1;
  for (auto iter : vchannelImg) {
    if (iter > v_max) {
      v_max = iter;
    }
    if (iter < v_min) {
      v_min = iter;
    }
  }
  // debug
  cout << "vmax: " << v_max << " vmin " << v_min << endl;
  for (int i = 0; i < row * column; i++) {
    vchannelImg[i] = (vchannelImg[i] - v_min) / (v_max - v_min);
  }

  // s
  vector<float> s(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    s[i] = log(255 * vchannelImg[i] + 1);
  }

  vector<float> l_curr(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    l_curr[i] = 1.05 * s[i];
    if (l_curr[i] < opt.epsilon_2) {
      l_curr[i] = opt.epsilon_2;
    }
  }

  vector<float> C{0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020};
  PU2Operator(C);
  // debug
  cout << "mask:" << endl;
  for (auto iter : C) {
    cout << iter << " ";
  }

  /* processing image */
  /* internal results */
  vector<float> Delta_l(row * column, 0.);
  vector<float> p_l(row * column, 0.);
  for (int t = 0; t <= opt.n; t++) {

    computeP(l_curr, s, C, p_l, row, column, 1);

    // debug
    cout << "computeP " << endl;

    for (int i = 0; i < row * column; i++) {

      if (-opt.v_3 < 0) {
        Delta_l[i] = p_l[i] * pow(opt.Delta_t, opt.v_3) -
                     2 * opt.mu / tgammaf(3 - opt.v_3) *
                         (Delta_l[i] * Delta_l[i]) * 1 /
                         (pow(l_curr[i], opt.v_3));
      } else {
        Delta_l[i] = p_l[i] * pow(opt.Delta_t, opt.v_3) -
                     2 * opt.mu / tgammaf(3 - opt.v_3) *
                         (Delta_l[i] * Delta_l[i]) * pow(l_curr[i], -opt.v_3);
      }

    } // Delta_l = p_l*(np.power(Delta_t,v_3)) - 2*mu/math.gamma(3 -
      // v_3)*(Delta_l*Delta_l) * (np.power(l_curr, (-v_3)))

    //  l_curr = l_curr + Delta_l
    for (int i = 0; i < row * column; i++) {
      l_curr[i] += Delta_l[i];
    }
    for (int i = 0; i < row * column; i++) {
      if (l_curr[i] < opt.epsilon_2) {
        l_curr[i] = opt.epsilon_2;
      }
    }
    // l_curr = np.where(l_curr <= 0, epsilon_2, l_curr)
  }

  for (int i = 0; i < row * column; i++) {
    l_curr[i] = exp(l_curr[i]);
  }
  // L = np.exp(l_curr)

  for (int i = 0; i < row * column; i++) {
    channels[2].data[i] =
        (unsigned char)(255 * (vchannelImg[i] /
                               pow(l_curr[i] / 255.0, 1 - 1 / opt.gamma_cor)));
  }

  // vChannel = (255 * vChannel/(np.power((L/255), (1 -
  // 1/gamma_corr)))).astype(np.uint8)
  cv::merge(channels, hsvImg);

  cv::Mat processedImg;
  cv::cvtColor(hsvImg, processedImg, cv::COLOR_HSV2RGB);
  cv::imwrite("./processImg.jpg", processedImg);
}

void utTestPrintVector(const vector<float> &vec, const string prefix) {

  if (!opt.debug) {
    return;
  }
  cout << prefix << ": " << endl;
  for (auto iter : vec) {
    cout << iter << " " << endl;
  }
  cout << "\n" << endl;
}

void utTestPad() {
  // test padding
  vector<float> img = {1.0, 2.0, 3, 4, 5, 6, 7, 8, 9};
  vector<float> paddedImg(81, 0.);
  iterativePad(img, paddedImg, 3, 4, 3);

  cout << "paddedImg:" << endl;
  for (int i = 0; i < 81; i++) {
    cout << paddedImg[i] << " ";
    if ((i + 1) % 9 == 0) {
      cout << "\n";
    }
  }
  cout << endl;
}

void utTestCal8Deriv() {
  int row = 3;
  int column = 3;
  int n = 5;
  vector<float> C{0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020};
  PU2Operator(C);
  vector<float> l = {1.0, 2.0, 3, 4, 5, 6, 7, 8, 9};
  /* calculate D8_l and D_l */
  /* allocate space to save D8 and D*/
  vector<vector<float>> D_l_8;
  for (int i = 0; i < 8; i++) {
    vector<float> temp(row * column, 0.);
    D_l_8.push_back(temp);
  }
  vector<float> D_l(row * column, 0.);
  cal8Deriv(l, D_l_8, D_l, C, row, column, 1);

  cout << "D_l: " << endl;
  for (int i = 0; i < row * column; i++) {
    cout << D_l[i] << " ";
  }
}

void utTestFstDiff() {
  int row = 4;
  int column = 3;
  vector<float> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  for (int i = 0; i < 8; i++) {
    vector<float> result(row * column, 0);
    fstDiff(A, result, i, row, column);

    cout << "result " << i << ": " << endl;
    for (auto iter : result) {
      cout << iter << " ";
    }
    cout << endl;
  }
}

void utTestComputeP() {
  int row = 3;
  int column = 3;
  vector<float> A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  /* prepare data */
  float v_min = 10000, v_max = -1;
  for (auto iter : A) {
    if (iter > v_max) {
      v_max = iter;
    }
    if (iter < v_min) {
      v_min = iter;
    }
  }

  // debug
  cout << "vmax: " << v_max << " vmin " << v_min << endl;

  for (int i = 0; i < row * column; i++) {
    A[i] = (A[i] - v_min) / (v_max - v_min);
  }
  utTestPrintVector(A, "A");

  // s
  vector<float> s(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    s[i] = log(255 * A[i] + 1.0);
  }

  // debug
  utTestPrintVector(s, "s");

  vector<float> l_curr(row * column, 0.);
  for (int i = 0; i < row * column; i++) {
    l_curr[i] = 1.05 * s[i];
  }
  utTestPrintVector(l_curr, "l_curr");
  for (auto iter : l_curr) {
    if (iter <= opt.epsilon_2) {
      iter = opt.epsilon_2;
    }
  }

  vector<float> C{0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020};
  PU2Operator(C);
  utTestPrintVector(C, "mask");

  vector<float> p_l(row * column, 0.);
  computeP(l_curr, s, C, p_l, row, column, 1);

  // print results
  utTestPrintVector(p_l, "p_l");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "usage: ./franctionalRetinex imgPath" << endl;
    return -1;
  }

  // utTest();
  // utTestCal8Deriv();
  utTestFstDiff();
  // utTestPad();

  // utTestComputeP();

  const char *imagepath = argv[1];
  cv::Mat m = cv::imread(imagepath, 1);
  if (m.empty()) {
    cout << "cv::imread " << imagepath << " failed" << endl;
    return -1;
  }

  // franctionalRetinex(m, m.rows, m.cols);

  return 0;
}