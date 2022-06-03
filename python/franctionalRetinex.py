import cv2
import numpy as np
import math

# 在这里设置参数
v_1 = 1.25
v_2 = 2.25
v_3 = 0.90
mu = 0.1
alpha_1 = 0.05
alpha_2 = 0.1
Delta_t = 0.002
n = 6
epsilon_1 = 0.006
epsilon_2 = 0.00001
gamma_corr = 2.2
N = 7
norm = 2

def PU2Operator(v, N):
    # PU2OPERATOR PU-2 分数阶微分算子掩模
    # 给定阶数 v 和掩模尺寸 N = n + 2，其中 N >= 3，N 通常为奇数
    # 返回掩模系数，从 C_{-1} 到 C_n

    # 生成一空的行向量，用于存储掩模系数
    n = N - 2
    coefficients = np.zeros(N) # 生成一空的行向量，用于存储掩模系数
    temp = np.array([v/4 + v*v/8, 1 - v*v/4, -v/4 + v*v/8])
    for k in range(-1, n-1):

        _k = k + epsilon_2

        coefficients[k + 1] = 1 / math.gamma(-v) * (math.gamma(_k - v + 1)/math.gamma(_k + 2)*temp[0] 
        + math.gamma(_k - v)/math.gamma(_k+1)*temp[1] 
        + math.gamma(_k - v - 1)/math.gamma(_k)*temp[2])


    coefficients[n] = math.gamma(n - v - 1)/math.gamma(n)/math.gamma(-v) * temp[1] 
    + math.gamma(n - v - 2)/math.gamma(n - 1)/math.gamma(-v) * temp[2]
    coefficients[n + 1] = math.gamma(n - v - 1)/math.gamma(n)/math.gamma(-v) * temp[2]

    return coefficients

def pad(img, n, row, column):
    # PAD 对图像矩阵的边缘进行拉格朗日插值延拓
    # img 为图像矩阵，n 为延拓次数，拉格朗日插值公式为 s(-1) = 3[s(0) - s(1)] + s(2)
    # row, column 为 A 的行数和列数

    # copy before padding
    paddedImg = np.zeros(shape=(row + n*2, column + n*2))
    for row in range(n, n + row):
        for col in range(n, n + column):
            paddedImg[row][column] = img[row-n][col-n]

    # padding
    for k in range(n):

        startRow, startCol = n - k, n -k
        endRow, endCol = row + k, col + k 
        
        # pad horizontally
        for r in range(startRow, endRow):
            # pad left
            paddedImg[r][startCol - 1] = 3*(paddedImg[r][startCol] - paddedImg[r][startCol + 1])
            + paddedImg[r][startCol + 2]

            # pad right
            paddedImg[r][endCol + 1] = 3*(paddedImg[r][endCol] - paddedImg[r][endCol - 1])
            + paddedImg[r][endCol - 2]
        
        # pad vertically
        for c in range(startCol, endCol):
            # pad top
            paddedImg[startRow - 1][c] = 3*(paddedImg[startRow][c] - paddedImg[startRow + 1][c])
            + paddedImg[startRow + 2][c]

            # pad bottom
            paddedImg[endRow + 1][c] = 3*(paddedImg[endRow][c] - paddedImg[endRow -1 ][c])
            + paddedImg[endRow - 2][c]
        
        # pad four corner
        # pad left up corner
        paddedImg[startRow - 1][startCol - 1] = 3*(paddedImg[startRow][startCol] - paddedImg[startRow + 1][startCol + 1])
        + paddedImg[startRow + 2][startCol + 2]
        # pad right up corner
        paddedImg[startRow - 1][endCol + 1] = 3*(paddedImg[startRow][endCol] - paddedImg[startRow + 1][endCol - 1])
        + paddedImg[startRow + 2][endCol - 2]
        # pad left down corner
        paddedImg[endRow + 1][startCol - 1] = 3*(paddedImg[endRow][startCol] - paddedImg[endRow - 1][startCol + 1])
        + paddedImg[endRow - 2][startCol + 2]
        # pad right down corner
        paddedImg[endRow + 1][endCol + 1] = 3*(paddedImg[endRow][endCol] - paddedImg[endRow - 1][endCol - 1])
        + paddedImg[endRow -2][endCol -2]

    return paddedImg

def cal8DerivTest():
    N = 7
    n = N - 2
    row, column = 3, 3
    mask = np.array([0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020])
    vchannelImgPad = np.array([
   [-19,   -18,   -17,   -16,   -15,   -14,   -13,   -12,   -11,   -11,   -11,   -11,   -11],
   [-16,   -15,   -14,   -13,   -12,   -11,   -10,    -9,    -8,    -8,    -8,    -8,    -8],
   [-13,   -12,   -11,   -10,    -9,    -8,    -7,    -6,    -5,    -5,    -5,    -5,    -5],
   [-10,    -9,    -8,    -7,    -6,    -5,    -4,    -3,    -2,    -2,    -2,    -2,    -2],
    [-7,    -6,    -5,    -4,    -3,    -2,    -1,     0,     1,     1,     1,     1,     1],
    [-4,    -3,    -2,    -1,     0,     1,     2,     3,     4,     4,     4,     4,     4],
    [-1,     0,     1,     2,     3,     4,     5,     6,     7,     7,     7,     7,     7],
     [2,     3,     4,     5,     6,     7,     8,     9,    10,    10,    10,    10,    10],
     [5,     6,     7,     8,     9,    10,    11,    12,    13,    13,    13,    13,    13],
     [5,     6,     7,     8,     9,    10,    11,    12,    13,    13,    13,    13,    13],
     [5,     6,     7,     8,     9,    10,    11,    12,    13,    13,    13,    13,    13],
     [5,     6,     7,     8,     9,    10,    11 ,   12,    13,    13,    13,    13,    13],
     [5,     6,     7,     8,     9,    10,    11,    12,    13,    13,    13,    13,    13]])

    D8 = np.zeros(shape=(8, row, column))
    # 分别计算八个方向的分数阶偏导数矩阵
    dTop = D8[0]
    for k in range(N):
        startRow, startCol = k, n
        endRow, endCol = startRow + row, startCol + column

        weight = mask[7-k-1]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    print("dTop:{}".format(dTop))
    
    dBottom = D8[1]
    for k in range(N):
        startRow, startCol = n - 1 + k, n
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dBottom[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    print("dBottom:{}".format(dBottom))

    dLeft = D8[2]
    for k in range(N):
        startRow, startCol = n, k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[N-k-1]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeft[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    print("dLeft: {}".format(dLeft))

    dRight = D8[3]
    for k in range(N):
        startRow, startCol = n, n - 1 + k
        endRow, endCol = startRow + row,  startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRight[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    print("dRight:{}".format(dRight))
    
    dLeftBotton = D8[4]
    for k in range(N):
        startRow, startCol = n + k - 1, n - k + 1
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeftBotton[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    print("dLeftBottom: {}".format(dLeftBotton))

    dRightTop = D8[5]
    for k in range(N):
        startRow, startCol = n + 1 - k, n -1 + k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRightTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]    
    
    print("dRightTop:{}".format(dRightTop))
    
    dLeftTop = D8[6]
    for k in range(N):
        startRow, startCol = n + 1 - k, n + 1 - k
        endRow, endCol = startRow + row, startCol + column
        
        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeftTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    print("dLeftTop:{}".format(dLeftTop))
    
    dRightDown = D8[7]
    for k in range(N):
        startRow, startCol = n - 1 + k, n - 1 + k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRightDown[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    print("dRightBottom:{}".format(dRightDown))

def cal8Deriv(vChanelImg, mask, norm=1):
    # CAL8DERIV 计算八个方向上的分数阶偏导数
    # 根据系数向量，对输入进来的矩阵计算八个方向上的偏导数矩阵
    # C 的长度为 n + 2
    # D8 的第三个维度的排列顺序为 u, d, l, r, ld, ru, lu, rd

    n = mask.shape[0] - 2
    row, column = vChanelImg.shape

    vchannelImgPad = pad(vChanelImg, n, row, column)

    # test
    #print("row:{}, column:{}, n:{}".format(row, column, n))

    D8 = np.zeros(shape=(8, row, column))
    # 分别计算八个方向的分数阶偏导数矩阵
    dTop = D8[0]
    for k in range(N):
        startRow, startCol = k, n
        endRow, endCol = startRow + row, startCol + column

        weight = mask[7-k-1]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    #print("dTop:{}".format(dTop))
    
    dBottom = D8[1]
    for k in range(N):
        startRow, startCol = n - 1 + k, n
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dBottom[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    #print("dBottom:{}".format(dBottom))

    dLeft = D8[2]
    for k in range(N):
        startRow, startCol = n, k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[N-k-1]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeft[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    #print("dLeft: {}".format(dLeft))

    dRight = D8[3]
    for k in range(N):
        startRow, startCol = n, n - 1 + k
        endRow, endCol = startRow + row,  startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRight[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    #print("dRight:{}".format(dRight))
    
    dLeftBotton = D8[4]
    for k in range(N):
        startRow, startCol = n + k - 1, n - k + 1
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeftBotton[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]
    
    #print("dLeftBottom: {}".format(dLeftBotton))

    dRightTop = D8[5]
    for k in range(N):
        startRow, startCol = n + 1 - k, n -1 + k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRightTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]    
    
    #print("dRightTop:{}".format(dRightTop))
    
    dLeftTop = D8[6]
    for k in range(N):
        startRow, startCol = n + 1 - k, n + 1 - k
        endRow, endCol = startRow + row, startCol + column
        
        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dLeftTop[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    #print("dLeftTop:{}".format(dLeftTop))
    
    dRightDown = D8[7]
    for k in range(N):
        startRow, startCol = n - 1 + k, n - 1 + k
        endRow, endCol = startRow + row, startCol + column

        weight = mask[k]
        for r in range(startRow, endRow):
            for c in range(startCol, endCol):
                dRightDown[r - startRow][c - startCol] += weight*vchannelImgPad[r][c]

    #print("dRightBottom:{}".format(dRightDown))

    D = np.zeros(shape=(row, column))
    # 计算全微分
    if norm == 1:
        D += np.abs(dTop) + np.abs(dBottom) + np.abs(dLeft) + np.abs(dRight) + np.abs(dLeftBotton) + np.abs(dRightTop) 
        + np.abs(dLeftTop) + np.abs(dRightDown); # 采用一范数的方法

    return D8, D


def fstDiffTest():
    row, column = 3,3
    A_pad = np.array([
    [-7,    -6,    -5,    -4,    -3,    -2,    -2],
    [-4,    -3,    -2,    -1,     0,     1,     1],
    [-1,     0,     1,     2,     3,     4,     4],
    [ 2,     3,     4,     5,     6,     7,     7],
    [ 5,     6,     7,     8,     9,    10,    10],
    [ 8,     9,    10,    11,    12,    13,    13],
    [ 8,     9,    10,    11,    12,    13,    13]
    ])

    mask = np.array([0.375, 0.375, -0.875, 0.125])
    n = 2
    direction = 8

    if direction == 1:
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = k, n
            endRow, endCol = startRow  + row, startCol + column

            weight = mask[4-k-1]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]

        print("result: {}".format(result))
        
    elif direction == 2: # D_d
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = 1 + k, n
            endRow, endCol = k + row + 1, 2 + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        
        print("result: {}".format(result))
    
    elif direction == 3: # D_l
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n, k
            endRow, endCol = n + row, column + k

            weight = mask[4-k-1]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]

        print("result: {}".format(result))
    
    elif direction == 4: # D_r
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n, n - 1 + k
            endRow, endCol = startRow + row,  startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        print("result: {}".format(result))
    
    elif direction == 5: # D_ld
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + k - 1, n - k + 1
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        print("result: {}".format(result))
    
    elif direction == 6: # D_ru
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + 1 - k, n -1 + k
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        print("result: {}".format(result))
    
    elif direction == 7: # D_lu
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + 1 - k, n + 1 - k
            endRow, endCol = startRow + row, startCol + column
            
            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        print("result: {}".format(result))
    else: # D_rd
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n - 1 + k, n - 1 + k
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        print("result: {}".format(result))
def fstDiff(A, direction):
    #FSTDIFF 对一个矩阵在指定方向上做一次差分,采用一阶分数阶微分方式
    #   对输入矩阵 A, 向 direction 方向做一次差分，采用一节分数阶微分方式
    # direction 设置如下:
    # [1, 2, 3, 4, 5, 6, 7, 8] := [D_u, D_d, D_l, D_r, D_ld, D_ru, D_lu, D_rd]

    row, column = A.shape

    mask = [0.375, 0.375, -0.875, 0.125]
    n = 2
    # A_pad = padarray(A, [1, 1], 'symmetric');
    A_pad = pad(A, 2, row, column)

    if direction == 1:
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = k, n
            endRow, endCol = startRow  + row, startCol + column

            weight = mask[4-k-1]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]

        #print("result: {}".format(result))
        
    elif direction == 2: # D_d
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = 1 + k, n
            endRow, endCol = k + row + 1, 2 + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        
        #print("result: {}".format(result))
    
    elif direction == 3: # D_l
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n, k
            endRow, endCol = n + row, column + k

            weight = mask[4-k-1]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]

        #print("result: {}".format(result))
    
    elif direction == 4: # D_r
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n, n - 1 + k
            endRow, endCol = startRow + row,  startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        #print("result: {}".format(result))
    
    elif direction == 5: # D_ld
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + k - 1, n - k + 1
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        #print("result: {}".format(result))
    
    elif direction == 6: # D_ru
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + 1 - k, n -1 + k
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        #print("result: {}".format(result))
    
    elif direction == 7: # D_lu
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n + 1 - k, n + 1 - k
            endRow, endCol = startRow + row, startCol + column
            
            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        #print("result: {}".format(result))
    else: # D_rd
        result = np.zeros(shape=(row, column))
        for k in range(4):
            startRow, startCol = n - 1 + k, n - 1 + k
            endRow, endCol = startRow + row, startCol + column

            weight = mask[k]
            for r in range(startRow, endRow):
                for c in range(startCol, endCol):
                    result[r - startRow][c - startCol] += weight*A_pad[r][c]
        #print("result: {}".format(result))
    
    return result


def computeP(v_1, v_2, v_3, alpha_1, alpha_2, l, s_0, epsilon_1, C, norm):
# computeP 根据当前矩阵 l^n，利用给定参数 α_1, α_2，计算 P(l^n)
# 针对当前的 l 矩阵，分别计算 l 和 (l - s) 在八个方向上的分数阶偏导数矩阵

    ls = l - s_0

    D_l_8, D_l = cal8Deriv(l, C, norm)
    D_l = np.where(D_l < epsilon_1, epsilon_1, D_l)

    [D_ls_8, D_ls] = cal8Deriv(ls, C, norm)
    D_ls = np.where(D_ls < epsilon_1, epsilon_1, D_ls)

    # 对 k 进行遍历，将求和符号逐 k 累加到 sum_k
    sum_k = 0
    for k in range(0, 2):
        prod_tau = 1
        for tau in range(2*k):
            prod_tau = prod_tau*(v_2 - tau + 1)

        # 计算在当前 k 值的情况下，八个方向的偏微分的差分的和
        eight_terms = 0
        D_l_power = np.power(D_l, (v_2 - 2*k - 2))
        D_ls_power = np.power(D_ls, (v_2 - 2*k - 2))
        second_term = alpha_1 * np.power((np.abs(ls)),(v_2 - 2*k - 2)) * ls

        for layer in range(8):
            temp = D_l_power * D_l_8[layer] + second_term + alpha_2 * D_ls_power * D_ls_8[layer]
            eight_terms = eight_terms + fstDiff(temp, layer)

        # do fractional derivative
        sum_k = sum_k + prod_tau/math.gamma(2*k + 1) * eight_terms

    P_l = -math.gamma(1 - v_1)/math.gamma(-v_1)/math.gamma(-v_3) * sum_k

    return P_l

def franctionalRetinex(imgPath):
    rgbImg = cv2.imread(imgPath)

    hsvImg = cv2.cvtColor(rgbImg,cv2.COLOR_BGR2HSV)
    _1, _2, vChannel = cv2.split(hsvImg)

    v_max, v_min = np.max(vChannel), np.min(vChannel)

    # test
    print("vchannel:{}, max:{}, min:{}".format(vChannel, v_max, v_min))

    vChannel = (vChannel - v_min)/(v_max - v_min)
    s = np.log(255*vChannel + 1)

    l_curr = 1.05*s
    l_curr = np.where(l_curr <= epsilon_2, epsilon_2, l_curr)
    
    Delta_l = 0
    C = PU2Operator(v_1, N)
    print(C)
    #C = np.array([0.5078, -0.0254, -0.7996, 0.2615, 0.0142, 0.0058, -0.0020])

    # test
    print("mask:{}".format(C))

    for t in range(1, n):
        p_l = computeP(v_1, v_2, v_3, alpha_1, alpha_2, l_curr, s, epsilon_1, C, norm=1)
        Delta_l = p_l*(np.power(Delta_t,v_3)) - 2*mu/math.gamma(3 - v_3)*(Delta_l*Delta_l) * (np.power(l_curr, (-v_3)))
        l_curr = l_curr + Delta_l
        
        l_curr = np.where(l_curr <= epsilon_2, epsilon_2, l_curr)
        
    L = np.exp(l_curr)
    
    # test
    print("vchannel:{}, other:{}".format(vChannel, np.power((L/255), (1 - 1/gamma_corr))))

    vChannel = (255 * vChannel/(np.power((L/255), (1 - 1/gamma_corr)))).astype(np.uint8)

    hsvImg = cv2.merge([_1, _2, vChannel])

    processedImg = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
    cv2.imwrite("./processImg.jpg", processedImg)

franctionalRetinex("./pot.jpg")
#cal8DerivTest()
#fstDiffTest()