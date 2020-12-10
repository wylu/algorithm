#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   归并排序.py
@Time    :   2020/11/27 13:43:28
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
归并排序（Merge sort），是创建在归并操作上的一种基于比较的排序算法。
1945 年由约翰·冯·诺伊曼首次提出。该算法是采用分治法（Divide and Conquer）
的一个非常典型的应用，且各层分治递归可以同时进行。

基本原理
归并排序算法是分治策略实现对 n 个元素进行排序的算法。

其基本思想是：将待排序元素分成大小大致相同的 2 个子集合，分别对 2 个
子集合进行排序，最终将排好序的子集合合并成为所要求的排好序的集合。

算法步骤
1.申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列；
2.设定两个指针，最初位置分别为两个已经排序序列的起始位置；
3.比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针
  到下一位置；
4.重复步骤 3 直到某一指针到达序列尾；
5.将另一序列剩下的所有元素直接复制到合并序列尾；
"""
from typing import List


def merge(nums: List[int], left: int, mid: int, right: int) -> None:
    tmp = [0] * (right - left + 1)
    i, j, k = left, mid + 1, 0
    while i <= mid or j <= right:
        if i > mid:
            tmp[k] = nums[j]
            j += 1
        elif j > right:
            tmp[k] = nums[i]
            i += 1
        else:
            if nums[i] <= nums[j]:
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
        k += 1

    for i in range(k):
        nums[left + i] = tmp[i]


def merge_sort(nums: List[int], left: int, right: int) -> None:
    if left < right:
        mid = (left + right) // 2
        merge_sort(nums, left, mid)
        merge_sort(nums, mid + 1, right)
        merge(nums, left, mid, right)


def merge_sort2(nums: List[int], left: int, right: int) -> None:
    size, n = 1, len(nums)
    while size < n:
        left = 0
        while left + size < n:
            mid = left + size - 1
            right = min(mid + size, n - 1)
            merge(nums, left, mid, right)
            left = right + 1
        size *= 2


if __name__ == "__main__":
    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    merge_sort(nums, 0, len(nums) - 1)
    print(nums)

    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    merge_sort2(nums, 0, len(nums) - 1)
    print(nums)
