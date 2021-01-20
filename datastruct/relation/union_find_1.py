#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   union_find_1.py
@Time    :   2020/09/18 09:05:01
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   并查集I
"""


class UnionFind:
    def __init__(self, n: int):
        self.par = list(range(n))

    def find(self, x: int) -> int:
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x: int, y: int) -> None:
        self.par[self.find(x)] = self.find(y)


if __name__ == '__main__':
    n = 10
    uf = UnionFind(n + 1)
    uf.union(1, 2)
    uf.union(1, 5)
    print(uf.find(2), uf.find(5))
    print(uf.find(2), uf.find(4))

    uf.union(6, 4)
    uf.union(4, 7)
    print(uf.find(6), uf.find(7))
    print(uf.find(2), uf.find(4))

    uf.union(5, 7)
    print(uf.find(2), uf.find(4))
