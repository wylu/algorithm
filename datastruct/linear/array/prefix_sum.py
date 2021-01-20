#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   prefix_sum.py
@Time    :   2020/09/20 21:26:27
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   前缀和
"""
from typing import List


def get_prefix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ps = [0] * (n + 1)
    for i in range(n):
        ps[i + 1] = ps[i] + nums[i]
    return ps


if __name__ == "__main__":
    print(get_prefix_sum([1, 2, 3, 4, 5]))
