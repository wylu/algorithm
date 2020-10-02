#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   模板I.py
@Time    :   2020/10/02 23:12:29
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
模板 #1 是二分查找的最基础和最基本的形式。这是一个标准的二分查找模板，
用于查找可以通过访问数组中的单个索引来确定的元素或条件。

关键属性
  - 二分查找的最基础和最基本的形式。
  - 查找条件可以在不与元素的两侧进行比较的情况下确定
   （或使用它周围的特定元素）。

区分语法
  - 初始条件：left = 0, right = length-1
  - 终止：left > right
  - 向左查找：right = mid-1
  - 向右查找：left = mid+1
"""


def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1


if __name__ == "__main__":
    print(binarySearch([-1, 0, 3, 5, 9, 12], 9))
    print(binarySearch([-1, 0, 3, 5, 9, 12], 2))
