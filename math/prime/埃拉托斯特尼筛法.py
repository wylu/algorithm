#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   埃拉托斯特尼筛法.py
@Time    :   2020/09/25 22:21:25
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
埃拉托斯特尼筛法

当一个数是素数的时候，它的倍数肯定不是素数，对于这些数可以直接标记筛除。

时间复杂度：O(n log log n)
"""
from typing import List


def eratosthenes_sieve(n: int) -> List[int]:
    ans = []
    marks = [True] * n
    for i in range(2, n):
        if marks[i]:
            ans.append(i)
            for j in range(i + i, n, i):
                marks[j] = False
    return ans


if __name__ == "__main__":
    print(eratosthenes_sieve(100))
