#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   模板III.py
@Time    :   2020/10/03 12:07:45
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
模板 #3 是二分查找的另一种独特形式。它用于搜索需要访问当前索引
及其在数组中的直接左右邻居索引的元素或条件。

关键属性
  - 实现二分查找的另一种方法。
  - 搜索条件需要访问元素的直接左右邻居。
  - 使用元素的邻居来确定它是向右还是向左。
  - 保证查找空间在每个步骤中至少有 3 个元素。
  - 需要进行后处理。当剩下 2 个元素时，循环 / 递归结束。
    需要评估其余元素是否符合条件。

区分语法
  - 初始条件：left = 0, right = length-1
  - 终止：left + 1 == right
  - 向左查找：right = mid
  - 向右查找：left = mid
"""


def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target:
        return left
    if nums[right] == target:
        return right
    return -1


if __name__ == "__main__":
    pass
