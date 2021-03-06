#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   quick_sort.py
@Time    :   2020/11/27 13:26:45
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   快速排序

快速排序（Quicksort），又称分区交换排序（partition-exchange sort），
简称快排，是一种高效的排序算法。由英国计算机科学家 Tony Hoare 于 1959
年开发并于 1961 年发表，至今它仍然是一种常用的排序算法。事实上，如果
实施得当，它可以比归并排序、堆排序快两到三倍。

基本原理
快速排序是图灵奖得主 C. R. A. Hoare 于 1960 年提出的一种划分交换排序，
它采用了一种分治的策略，通常称其为 分治法(Divide-and-ConquerMethod)。

分治法的基本思想是：将原问题分解为若干个规模更小但结构与原问题相似的
子问题。递归地求解这些子问题，然后将这些子问题的解组合为原问题的解。

算法步骤
1.从序列中挑出一个元素，作为"基准" (pivot)
2.把所有比基准值小的元素放在基准前面，所有比基准值大的元素放在基准值
  的后面（相同的数可以放到任一边），这个操作称为分区 (partition) 操作，
  分区操作结束后，基准元素所处的位置就是最终排序后它的位置
3.递归地把小于基准值元素的子数列和大于基准值元素的子数列排序，直到所有
  子序列的大小为 0 或 1，这时整体已经排好序了
"""
from typing import List


def partition(nums: List[int], left: int, right: int) -> None:
    j = left - 1
    for i in range(left, right):
        if nums[i] <= nums[right]:
            j += 1
            nums[i], nums[j] = nums[j], nums[i]
    j += 1
    nums[j], nums[right] = nums[right], nums[j]
    return j


def quick_sort(nums: List[int], left: int, right: int) -> None:
    if left < right:
        idx = partition(nums, left, right)
        quick_sort(nums, left, idx - 1)
        quick_sort(nums, idx + 1, right)


if __name__ == "__main__":
    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    quick_sort(nums, 0, len(nums) - 1)
    print(nums)
