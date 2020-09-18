#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   快慢双指针.py
@Time    :   2020/09/15 19:37:47
@Author  :   wylu
@Version :   1.0
@Contact :   15wylu@gmail.com
@License :   Copyright © 2020, wylu-CHINA-SHENZHEN. All rights reserved.
@Desc    :
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


def sf(head: ListNode) -> bool:
    # Initialize slow & fast pointers
    slow, fast = head, head

    # Change this condition to fit specific problem.
    # Attention: remember to avoid null-pointer error
    while fast and fast.next:
        slow = slow.next  # move slow pointer one step each time
        fast = fast.next.next  # move fast pointer two steps each time
        # change this condition to fit specific problem
        if slow == fast:
            return True

    # change return value to fit specific problem
    return False


if __name__ == '__main__':

    def mk_alist(*args):
        dummy = ListNode(0)
        cur = dummy
        for val in args:
            cur.next = ListNode(val)
            cur = cur.next
        return dummy.next

    def to_list(head):
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res

    head = mk_alist(1, 2, 3, 4, 5)
    # print(to_list(head))

    # make a ring
    cur, node3 = head, None
    for i in range(4):
        if i == 2:
            node3 = cur
        cur = cur.next
    cur.next = node3

    print(sf(head))
