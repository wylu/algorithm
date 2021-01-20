#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   euler_linear_sieve.py
@Time    :   2020/09/25 22:43:21
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   欧拉线性筛法

基本思路
任意一个合数（2 不是合数），都可以表示成素数的乘积。

每个合数必有一个最小素因子，如果每个合数都用最小素因子筛去，那个这个合数就
不会被重复标记筛去，所以算法为线性时间复杂度。

例如合数 30 = 2 * 3 * 5 ，这个合数一定是被最小素因子 2 筛去的。

时间复杂度：O(n)
"""
from typing import List


def euler_sieve(n: int) -> List[int]:
    ans = []
    # True 表示该下标值为素数
    marks = [True] * n
    for i in range(2, n):
        if marks[i]:
            ans.append(i)

        j = 0
        while j < len(ans) and i * ans[j] < n:
            marks[i * ans[j]] = False
            if i % ans[j] == 0:
                break
            j += 1
    return ans


if __name__ == "__main__":
    print(euler_sieve(100))
