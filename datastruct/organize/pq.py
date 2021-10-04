#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   pq.py
@Time    :   2021/10/04 14:56:50
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
"""
from typing import Any
from typing import Callable
from typing import Iterable


class Heap:
    def __init__(self, cmp: Callable = None) -> None:
        self._cmp = cmp if cmp else self.minimum
        self._data = []
        self._size = 0

    @classmethod
    def heapify(cls, objs: Iterable, cmp: Callable = None) -> 'Heap':
        heap = Heap(cmp)
        for obj in objs:
            heap.push(obj)
        return heap

    @classmethod
    def minimum(cls, obj1: Any, obj2: Any) -> bool:
        return obj1 <= obj2

    @classmethod
    def maximum(cls, obj1: Any, obj2: Any) -> bool:
        return obj1 >= obj2

    def _sinking(self):
        if self._size <= 1:
            return

        obj = self._data[0]
        x = 0
        while x * 2 + 1 < self._size:
            a, b = x * 2 + 1, x * 2 + 2
            if b < self._size and self._cmp(self._data[b], self._data[a]):
                a = b

            if self._cmp(obj, self._data[a]):
                break

            self._data[x] = self._data[a]
            x = a

        self._data[x] = obj

    def _floating(self):
        obj = self._data[self._size]
        x = self._size
        while x:
            y = (x - 1) // 2
            if self._cmp(self._data[y], obj):
                break

            self._data[x] = self._data[y]
            x = y

        self._data[x] = obj

    def push(self, obj: Any) -> None:
        self._data.append(obj)
        self._floating()
        self._size += 1

    def pop(self) -> Any:
        obj = self._data[0]
        self._size -= 1
        self._data[0] = self._data[self._size]
        self._sinking()
        self._data.pop()
        return obj

    def front(self, default: Any = None) -> Any:
        return self._data[0] if self._size else default

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> 'Heap':
        return self

    def __next__(self) -> Any:
        if len(self) == 0:
            raise StopIteration
        return self.pop()


if __name__ == '__main__':
    import random
    nums = [random.randint(1, 100) for _ in range(10)]
    # pq = Heap.heapify(nums)
    pq = Heap.heapify(nums, cmp=Heap.maximum)
    # print(pq._data)

    print(nums)
    print([num for num in pq])
