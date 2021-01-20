#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   suffix_sum.py
@Time    :   2020/09/20 21:30:08
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   后缀和
"""
from typing import List


def get_suffix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ss = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        ss[i] = ss[i + 1] + nums[i]
    return ss


if __name__ == "__main__":
    print(get_suffix_sum([5, 4, 3, 2, 1]))
