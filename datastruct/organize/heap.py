#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   heap.py
@Time    :   2020/11/27 14:25:14
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   堆

堆（Heap）是计算机科学中的一种特别的树状数据结构。若是满足以下特性，
即可称为堆：“给定堆中任意节点 P 和 C，若 P 是 C 的父节点，那么 P
的值会小于等于（或大于等于） C 的值”。

堆始于 J._W._J._Williams 在 1964 年发表的堆排序（heap sort），
当时他提出了二叉堆树作为此算法的数据结构。堆在 Dijkstra 等几种
有效的图形算法中也非常重要。

堆是一种称为优先级队列的抽象数据类型的最有效实现，实际上优先级队列
通常被称为“堆”，而不管它们如何实现。

严格来说，堆也有不同的种类。二叉堆本质上是完全二叉树，可以分为两种类型：
  - 最大堆（大顶堆）
  - 最小堆（小顶堆）

二叉堆的根节点叫做 堆顶。最大堆和最小堆的特点，决定了在最大堆的堆顶
是整个堆中的 最大元素；最小堆的堆顶是整个堆中的 最小元素。

堆的逻辑结构与物理存储
二叉堆在逻辑上虽然是一颗完全二叉树，但它的存储方式并不是链式存储，
而是顺序存储。换句话说，二叉堆的所有节点都存储在数组当中。

树的节点是按从上到下、从左到右的顺序紧凑排列的。

利用数组下标作为节点编号，假设父节点的下标是 parent，则有
  - 左儿子的下标 = 2 * parent + 1
  - 右儿子的下标 = 2 * parent + 2
"""


class MinimumHeap:
    def __init__(self):
        self._heap = []
        self._size = 0

    def _floating(self, i: int, x: int) -> None:
        """上浮

        Args:
            i (int): 插入节点的索引
            x (int): 插入节点
        """
        while i > 0:
            # 父节点的索引
            p = (i - 1) // 2
            # 如果已经没有大小颠倒则退出
            if self._heap[p] <= x:
                break
            # 自己上浮
            self._heap[i] = self._heap[p]
            i = p
        self._heap[i] = x

    def _sinking(self, x: int) -> None:
        """下沉

        Args:
            x (int): 堆顶节点
        """
        i = 0
        while i * 2 + 1 < self._size:
            # 比较子节点的值
            a, b = i * 2 + 1, i * 2 + 2
            if b < self._size and self._heap[b] < self._heap[a]:
                a = b
            # 如果已经没有大小颠倒则退出
            if self._heap[a] >= x:
                break
            # 自己下沉
            self._heap[i] = self._heap[a]
            i = a
        self._heap[i] = x

    def __len__(self):
        return self._size

    def push(self, x: int) -> None:
        self._heap.append(0)
        self._floating(self._size, x)
        self._size += 1

    def pop(self) -> int:
        ans = self._heap[0]
        self._sinking(self._heap[self._size - 1])
        self._size -= 1
        return ans

    def top(self) -> int:
        return self._heap[0]


if __name__ == "__main__":
    heap = MinimumHeap()
    for num in [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]:
        heap.push(num)

    print([heap.pop() for _ in range(len(heap))])
