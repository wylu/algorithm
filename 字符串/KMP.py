#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   KMP.py
@Time    :   2020/09/18 09:51:15
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
KMP 利用之前已经部分匹配这个有效信息，保持 i 不回溯，通过修改 j 的位置，
让模式串尽量地移动到有效的位置，而模式串移动的信息存在 next 数组中。

KMP 算法的关键是 next 数组的计算，next 数组的计算只与模式串有关，
next 中存储的值为当前下标之前的子串的 最长相同前缀和后缀的长度。
如模式串 "ABCDABD" 的 next 数组：

    i         0  1  2  3  4  5  6
    pattern   A  B  C  D  A  B  D
    next[i]  -1	 0  0  0  0  1  2

注：next[0] 设为 -1

已知 next[j] = k，如何求出 next[j+1]？
  - 如果 p[j] == p[k] , 则 next[j+1] = next[j] + 1 = k + 1;
  - 如果 p[j] != p[k], 则令 k = next[k], 如果此时 p[j] == p[k],
    则 next[j+1] = k+1, 如果不相等, 则继续递归前缀索引,
    令 k = next[k], 继续判断, 直至 k = -1 (即 k = next[0] )
    或者 p[j] == p[k] 为止
"""
from typing import List


# 计算失配指针
def cal_fails(p: str) -> List[int]:
    k, j, n = -1, 0, len(p)
    fails = [-1] * n
    while j < n - 1:
        if k == -1 or p[k] == p[j]:
            k += 1
            j += 1
            fails[j] = k
        else:
            k = fails[k]
    return fails


# 匹配
def search(s: str, p: str) -> int:
    fails = cal_fails(p)
    i, j, sl, pl = 0, 0, len(s), len(p)
    while i < sl and j < pl:
        if j == -1 or s[i] == p[j]:
            i += 1
            j += 1
        else:
            j = fails[j]
    return i - j if j == pl else -1


if __name__ == '__main__':
    pattern = 'ABCDABD'
    fails = cal_fails(pattern)
    print(fails)

    source = 'BBC ABCDAB ABCDABCDABDE'
    start = search(source, pattern)
    end = start + len(pattern)

    for ch in source:
        print('{: >2}'.format(ch), end=' ')
    print()
    for i in range(len(source)):
        print('{: >2}'.format(i), end=' ')
    print()

    print(start, end, source[start:end])
