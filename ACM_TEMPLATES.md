[TOC]

# ACM 算法模板

# 图论

## 路径问题

### 0/1边权最短路径

### BFS

### 非负边权最短路径（Dijkstra）

可以用Dijkstra解决问题的特征

### 负边权最短路径

### Bellman-Ford

#### Bellman-Ford的Yen-氏优化

#### 差分约束系统

### Floyd

#### 广义路径问题

#### 传递闭包

#### 极小极大距离 / 极大极小距离

### Euler Path / Tour

#### 圈套圈算法

#### 混合图的 Euler Path / Tour

### Hamilton Path / Tour

#### 特殊图的Hamilton Path / Tour 构造

## 生成树问题

### 最小生成树

### 第k小生成树

### 最优比率生成树

### 0/1分数规划

### 度限制生成树

## 连通性问题

### 强大的DFS算法

### 无向图连通性

#### 割点

#### 割边

#### 二连通分支

#### 有向图连通性

#### 强连通分支

#### 2-SAT

#### 最小点基

## 有向无环图

### 拓扑排序

#### 有向无环图与动态规划的关系

## 二分图匹配问题

### 一般图问题与二分图问题的转换思路

### 最大匹配

#### 有向图的最小路径覆盖

#### 0 / 1矩阵的最小覆盖

### 完备匹配

### 最优匹配

### 稳定婚姻

## 网络流问题

### 网络流模型的简单特征和与线性规划的关系

### 最大流最小割定理

### 最大流问题

#### 有上下界的最大流问题

### 循环流

### 最小费用最大流 / 最大费用最大流

## 弦图的性质和判定

# 组合数学

## 解决组合数学问题时常用的思想

### 逼近

### 递推 / 动态规划

## 概率问题

### Polya定理

# 计算几何 / 解析几何

## 计算几何的核心：叉积 / 面积

## 解析几何的主力：复数

## 基本形

### 点

### 直线，线段

### 多边形

## 凸多边形 / 凸包

### 凸包算法的引进，卷包裹法

## Graham扫描法

### 水平序的引进，共线凸包的补丁

## 完美凸包算法

## 相关判定

### 两直线相交

### 两线段相交

### 点在任意多边形内的判定

### 点在凸多边形内的判定

## 经典问题

### 最小外接圆

#### 近似O(n)的最小外接圆算法

### 点集直径

#### 旋转卡壳，对踵点

### 多边形的三角剖分

# 数学 / 数论

## 最大公约数

### Euclid算法

#### 扩展的Euclid算法

### 同余方程 / 二元一次不定方程

### 同余方程组

## 线性方程组

### 高斯消元法

#### 解mod 2域上的线性方程组

### 整系数方程组的精确解法

## 矩阵

### 行列式的计算

#### 利用矩阵乘法快速计算递推关系

## 分数

### 分数树

### 连分数逼近

## 数论计算

### 求N的约数个数

### 求phi(N)

### 求约数和

### 快速数论变换

## 素数问题

### 概率判素算法

### 概率因子分解

# 数据结构

## 线性结构

### 链表

链表结点定义

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
```

#### 快慢双指针

```python
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
```

#### 快慢指针判断链表是否有环

```python
def hasCycle(self, head: ListNode) -> bool:
	if head:
		slow, fast = head, head
		while fast and fast.next:
			slow = slow.next
			fast = fast.next.next
			if slow == fast:
				return True

	return False
```

#### 求环的入口结点

Floyd 算法被划分成两个不同的 阶段 。在第一阶段，找出列表中是否有环，如果没有环，可以直接返回 null 并退出。否则，用 相遇节点 来找到环的入口。

- 阶段 1

  这里我们初始化两个指针 - 快指针和慢指针。我们每次移动慢指针一步、快指针两步，直到快指针无法继续往前移动。如果在某次移动后，快慢指针指向了同一个节点，我们就返回它。否则，我们继续，直到 while 循环终止且没有返回任何节点，这种情况说明没有成环，我们返回 null 。

- 阶段 2

  给定阶段 1 找到的相遇点，阶段 2 将找到环的入口。首先我们初始化额外的两个指针： ptr1 ，指向链表的头， ptr2 指向相遇点。然后，我们每次将它们往前移动一步，直到它们相遇，它们相遇的点就是环的入口，返回这个节点。

```python
def hasCycle(head: ListNode) -> ListNode:
	if head:
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
```

#### 求两个单链表相交的起始节点

```python
def getIntersectionNode(self, ha: ListNode, hb: ListNode) -> ListNode:
    ca, cb = ha, hb
    while ca != cb:
        ca = ca.next if ca else hb
        cb = cb.next if cb else ha
    return ca
```

## 组织结构

### 二叉堆

### 左偏树

### 二项树

### 胜者树

### 跳跃表

### 样式图标

### 斜堆

### reap

## 统计结构

### 树状数组

### 虚二叉树

### 线段树

#### 矩形面积并

#### 圆形面积并

## 关系结构

### Hash表

### 并查集

#### 路径压缩思想的应用

## STL中的数据结构

### vector

### deque

### set / map

# 动态规划 / 记忆化搜索

## 动态规划和记忆化搜索在思考方式上的区别

## 最长子序列系列问题

### 最长不下降子序列

### 最长公共子序列

### 最长公共不下降子序列

## 一类NP问题的动态规划解法

## 树型动态规划

## 背包问题

## 动态规划的优化

### 四边形不等式

### 函数的凸凹性

### 状态设计

### 规划方向

# 线性规划

# 常用思想

## 二分

## 最小表示法

# 串

## KMP

## Trie结构

## 后缀树/后缀数组

## LCA/RMQ

## 有限状态自动机理论

# 排序

## 选择/冒泡

## 快速排序

## 堆排序

## 归并排序

## 基数排序

## 拓扑排序

## 排序网络