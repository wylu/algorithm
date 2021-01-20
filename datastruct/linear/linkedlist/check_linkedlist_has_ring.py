#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   check_linkedlist_has_ring.py
@Time    :   2021/01/20 22:55:36
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   快慢指针判断链表是否有环
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


def hasCycle(self, head: ListNode) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
