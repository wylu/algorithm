#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   matrix_transpose.py
@Time    :   2020/10/02 00:11:53
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   矩阵的转置

1, 2, 3        1, 4, 7        1, 4, 7
4, 5, 6   ->   2, 5, 6   ->   2, 5, 8
7, 8, 9        3, 8, 9        3, 6, 9
"""
from typing import List


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(transpose(matrix))
