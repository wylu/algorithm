#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   模板II.py
@Time    :   2020/10/03 10:49:39
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
模板 #2 是二分查找的高级模板。它用于查找需要访问数组中当前索引
及其直接右邻居索引的元素或条件。

关键属性
  - 一种实现二分查找的高级方法。
  - 查找条件需要访问元素的直接右邻居。
  - 使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
  - 保证查找空间在每一步中至少有 2 个元素。
  - 需要进行后处理。当你剩下 1 个元素时，循环 / 递归结束。
    需要评估剩余元素是否符合条件。
 
区分语法
  - 初始条件：left = 0, right = length
  - 终止：left == right
  - 向左查找：right = mid
  - 向右查找：left = mid+1
"""


def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1


# 可用于搜索左边界或插入位置
def searchLeftMargin(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


# 可用于搜索右边界或插入位置
def searchRightMargin(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left


if __name__ == "__main__":
    print(binarySearch([1, 2, 3, 4, 5], 3))

    print('\n================= 搜索左边界 =================')
    print(searchLeftMargin([5, 7, 7, 8, 8, 10], 8))
    print(searchLeftMargin([5, 7, 7, 8, 8, 10], 6))
    print(searchLeftMargin([7, 7, 8, 8], 7))
    print(searchLeftMargin([7, 7, 8, 8], 8))

    print('\n================= 搜索右边界 =================')
    print(searchRightMargin([5, 7, 7, 8, 8, 10], 8))
    print(searchRightMargin([5, 7, 7, 8, 8, 10], 6))
    print(searchRightMargin([7, 7, 8, 8], 7))
    print(searchRightMargin([7, 7, 8, 8], 8))
