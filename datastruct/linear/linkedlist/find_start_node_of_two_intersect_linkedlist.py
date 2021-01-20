#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   find_start_node_of_two_intersect_linkedlist.py
@Time    :   2021/01/20 22:59:42
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   求两个单链表相交的起始节点
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


def getIntersectionNode(self, ha: ListNode, hb: ListNode) -> ListNode:
    ca, cb = ha, hb
    while ca != cb:
        ca = ca.next if ca else hb
        cb = cb.next if cb else ha
    return ca
