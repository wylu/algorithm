#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   find_entry_point_of_ring.py
@Time    :   2021/01/20 22:57:04
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :   寻找环的入口结点

Floyd 算法被划分成两个不同的 阶段 。在第一阶段，找出列表中是否有环，
如果没有环，可以直接返回 null 并退出。否则，用 相遇节点 来找到环的入口。

- 阶段 1
  这里我们初始化两个指针 - 快指针和慢指针。我们每次移动慢指针一步、
  快指针两步，直到快指针无法继续往前移动。如果在某次移动后，快慢指针
  指向了同一个节点，我们就返回它。否则，我们继续，直到 while 循环
  终止且没有返回任何节点，这种情况说明没有成环，我们返回 null 。

- 阶段 2
  给定阶段 1 找到的相遇点，阶段 2 将找到环的入口。首先我们初始化
  额外的两个指针： ptr1 指向链表的头， ptr2 指向相遇点。然后，我们
  每次将它们往前移动一步，直到它们相遇，它们相遇的点就是环的入口，
  返回这个节点。
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


def hasCycle(head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return slow


def detectCycle(head: ListNode) -> ListNode:
    fast = hasCycle(head)
    if fast:
        slow = head
        while fast != slow:
            slow = slow.next
            fast = fast.next
        return slow
