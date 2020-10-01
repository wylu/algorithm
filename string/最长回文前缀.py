#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   最长回文前缀.py
@Time    :   2020/10/01 11:21:18
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
给定一个字符串 s，求其最长回文前缀的长度？

KMP算法

问题可转化为求字符串的最长前缀回文串，使用 KMP 字符串匹配算法可以避免
待匹配的串回退，提高查找速度。

具体地，记 s' 为 s 的反序，由于 s1 是 s 的前缀，那么 s1' 就是 s' 的后缀。
考虑到 s1 是一个回文串，因此 s1 = s1'，s1 同样是 s' 的后缀。

这样一来，我们将 s 作为模式串，s' 作为查询串进行匹配。当遍历到 s' 的末尾时，
如果匹配到 s 中的第 i 个字符，那么说明 s 的前 i 个字符与 s' 的后 i 个字符
相匹配（即相同），s 的前 i 个字符对应 s1，s' 的后 i 个字符对应 s1'，由于
s1 = s1'，因此 s1 就是一个回文串。

如果存在更长的回文串，那么 KMP 算法的匹配结果也会大于 i，
因此 s1 就是最长的前缀回文串。

示例 1：（无回退）

s  =  abbacd    s1  = abba
s' =  dcabba    s1' = abba

fail = [-1, 0, 0, 0, 1, 0], best = -1

i = 0, s'[0] = d, s[best+1] = s[0] = a, s'[0] != s[0], best = -1
i = 1, s'[1] = c, s[best+1] = s[0] = a, s'[1] != s[0], best = -1
i = 2, s'[2] = a, s[best+1] = s[0] = a, s'[2] == s[0], best = best+1 = 0
i = 3, s'[3] = b, s[best+1] = s[1] = b, s'[3] == s[1], best = best+1 = 1
i = 4, s'[4] = b, s[best+1] = s[2] = b, s'[4] == s[2], best = best+1 = 2
i = 5, s'[5] = a, s[best+1] = s[3] = a, s'[5] == s[3], best = best+1 = 3

示例 2：（有回退）

s  =  ababxybaba    s1  =  aba
s' =  ababyxbaba    s1' =  aba

fail = [-1, 0, 0, 1, 2, 0, 0, 0, 1, 2], best = -1

匹配到 x 处发生回退：

s  =  abab x ybaba
s' =  abab y xbaba

回退到第二个 a 处：

s  =    ab a bxybaba
s' =  abab y xbaba
"""


def get_longest_palindrome_prefix(s: str) -> int:
    # 计算失配指针（next 数组）
    n = len(s)
    fail = [-1] * n
    k, j = -1, 0
    while j < n - 1:
        if k == -1 or s[k] == s[j]:
            k += 1
            j += 1
            fail[j] = k
        else:
            k = fail[k]

    best = -1
    for i in range(n - 1, -1, -1):
        while best != -1 and s[best + 1] != s[i]:
            best = fail[best]
        if s[best + 1] == s[i]:
            best += 1

    return best + 1


def get_longest_palindrome_suffix(s: str) -> int:
    return get_longest_palindrome_prefix(s[::-1])


if __name__ == "__main__":
    glpp = get_longest_palindrome_prefix
    print(glpp('abbacd'))
    print(glpp('ababxybaba'))
    print(glpp('aacecaaa'))

    glps = get_longest_palindrome_suffix
    print(glps('dcabba'))
    print(glps('ababyxbaba'))
    print(glps('aaacecaa'))
