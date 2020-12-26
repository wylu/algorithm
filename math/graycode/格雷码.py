#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   格雷码.py
@Time    :   2020/12/26 10:50:25
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数
的差异。格雷编码序列必须以 0 开头。

方法一：
格雷编码：
设G(n)表示总位数为n的各类编码集合，根据以下策略可以求出G(n+1)
1. 将G(n)的每个元素前添加0得到G'(n);
2. 将G'(n)反转得到镜像R(n)，在R(n)中的每个元素前添加1得到R'(n);
3. 将G'(n)与R'(n)合并得到G(n+1);

编码思路：
1. 初始化G(0)和位数标识base;
2. 外层循环次数为总位数n;
3. 内层循环倒序遍历res,位数标识加上当前索引对应的值,
   即为R'(n)中的元素;
4. 在res后追加上述计算的元素，遍历结束，得到Gray编码集;


方法二：
二进制转格雷码公式：

某二进制数：         B[n-1],B[n-2],...,B[2],B[1],B[0]
其对应的格雷码：      G[n-1],G[n-2],...,G[2],G[1],G[0]
其中：最高位保留      G[n-1] = B[n-1]
     其它各位        G[i] = B[i+1] ^ B[i],  i = 0, 1, 2, ..., n-2

https://pic.leetcode-cn.com/1013850d7f6c8cf1d99dc0ac3292264b74f6a52d84e0215f540c80952e184f41-image.png

例如：二进制数为      1  0  1  1  0
                   |\ |\ |\ |\ |  # noqa W605
                   | \| \| \| \|  # noqa W605
                   |  ^  ^  ^  ^
                   |  |  |  |  |
     格雷码为：      1  1  1  0  1

遍历 0 到 2^(n-1)，利用公式转换，即最高位保留，其它位是当前位和
它的高一位进行异或操作。
"""
from typing import List


def binary2graycode(n: int) -> int:
    return n ^ (n >> 1)


def gen_gray_code_seq(n: int) -> List[int]:
    ans, base = [0], 1
    for i in range(n):
        for j in range(len(ans) - 1, -1, -1):
            ans.append(base + ans[j])
        base <<= 1
    return ans


def gen_gray_code_seq2(n: int) -> List[int]:
    ans = []
    start, end = 0, 1 << n
    while start < end:
        ans.append(start ^ (start >> 1))
        start += 1
    return ans


if __name__ == "__main__":
    print(gen_gray_code_seq(3))
    print(gen_gray_code_seq2(3))

    print(gen_gray_code_seq(4))
    print(gen_gray_code_seq2(4))

    print(binary2graycode(2))  # 2(10) => 3(11)
    print(binary2graycode(3))  # 3(11) => 2(10)
    print(binary2graycode(6))  # 6(110) => 5(101)
