#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   fast_power.py
@Time    :   2020/10/04 23:54:10
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   快速幂

https://oi-wiki.org/math/quick-pow/
"""


def quick_pow(a, b):
    if b < 0:
        b = -b
        a = 1 / a

    ans = 1
    while b > 0:
        if b & 1 == 1:
            ans *= a
        a *= a
        b >>= 1

    return ans


if __name__ == "__main__":
    print(quick_pow(2, 10))
    print(quick_pow(2.0, 10))
    print(quick_pow(2.0, -10))
