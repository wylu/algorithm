#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   gcd.py
@Time    :   2020/10/26 23:18:46
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   最大公约数 Greatest common divisor
"""


def gcd(x: int, y: int) -> int:
    return x if y == 0 else gcd(y, x % y)


if __name__ == "__main__":
    print(gcd(12, 4))
    print(gcd(12, 3))
