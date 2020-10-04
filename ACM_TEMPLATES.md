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

### 埃拉托斯特尼筛法

当一个数是素数的时候，它的倍数肯定不是素数，对于这些数可以直接标记筛除。

时间复杂度：O(n log log n)

```python
from typing import List


def eratosthenes_sieve(n: int) -> List[int]:
    ans = []
    marks = [True] * n
    for i in range(2, n):
        if marks[i]:
            ans.append(i)
            for j in range(i + i, n, i):
                marks[j] = False
    return ans
```

### 欧拉线性筛法

基本思路

任意一个合数（2 不是合数），都可以表示成素数的乘积。

每个合数必有一个最小素因子，如果每个合数都用最小素因子筛去，那个这个合数就

不会被重复标记筛去，所以算法为线性时间复杂度。

例如合数 30 = 2 * 3 * 5 ，这个合数一定是被最小素因子 2 筛去的。

时间复杂度：O(n)

```python
from typing import List


def euler_sieve(n: int) -> List[int]:
    ans = []
    # True 表示该下标值为素数
    marks = [True] * n
    for i in range(2, n):
        if marks[i]:
            ans.append(i)

        j = 0
        while j < len(ans) and i * ans[j] < n:
            marks[i * ans[j]] = False
            if i % ans[j] == 0:
                break
            j += 1
    return ans
```

### 概率判素算法

### 概率因子分解

## 快速幂

[https://oi-wiki.org/math/quick-pow/](https://oi-wiki.org/math/quick-pow/)

```python
def quick_pow(a, b):
    if b < 0:
        b = -b
        a = 1 / a

    ans = 1
    while b > 0:
        if b & 1 == 1:
            ans *= a
        a *= a
        b >>= 1

    return ans
```

# 数据结构

## 线性结构

### 二分查找

#### 模板 I

模板 1 是二分查找的最基础和最基本的形式。这是一个标准的二分查找模板，用于查找可以通过访问数组中的单个索引来确定的元素或条件。

**关键属性**

- 二分查找的最基础和最基本的形式。
- 查找条件可以在不与元素的两侧进行比较的情况下确定（或使用它周围的特定元素）。

**区分语法**

- 初始条件：left = 0, right = length-1
- 终止：left > right
- 向左查找：right = mid-1
- 向右查找：left = mid+1

```python
def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
```

#### 模板 II

模板 #2 是二分查找的高级模板。它用于查找需要访问数组中当前索引及其直接右邻居索引的元素或条件。

**关键属性**

- 一种实现二分查找的高级方法。
- 查找条件需要访问元素的直接右邻居。
- 使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
- 保证查找空间在每一步中至少有 2 个元素。
- 需要进行后处理。当你剩下 1 个元素时，循环 / 递归结束。需要评估剩余元素是否符合条件。

**区分语法**

- 初始条件：left = 0, right = length
- 终止：left == right
- 向左查找：right = mid
- 向右查找：left = mid+1

```python
def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1
```

##### 可用于搜索左边界或插入位置

```python
# 可用于搜索左边界或插入位置
def searchLeftMargin(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

##### 可用于搜索右边界或插入位置

```python
def searchRightMargin(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if nums[mid] <= target:
            left = mid
        else:
            right = mid - 1
    return left
```

#### 模板 III

### 数组

#### 前缀和

```python
def get_prefix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ps = [0] * (n + 1)
    for i in range(1, n + 1):
        ps[i] = ps[i - 1] + nums[i - 1]
    return ps
```

#### 后缀和

```python
def get_suffix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ss = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        ss[i] = ss[i + 1] + nums[i]
    return ss
```

#### 矩阵的转置

```python
"""
1, 2, 3        1, 4, 7        1, 4, 7
4, 5, 6   ->   2, 5, 6   ->   2, 5, 8
7, 8, 9        3, 8, 9        3, 6, 9
"""
from typing import List


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix
```

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

  给定阶段 1 找到的相遇点，阶段 2 将找到环的入口。首先我们初始化额外的两个指针： ptr1 指向链表的头， ptr2 指向相遇点。然后，我们每次将它们往前移动一步，直到它们相遇，它们相遇的点就是环的入口，返回这个节点。

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

#### Template1

```python
class UnionFind:
    def __init__(self, n: int):
        self.par = list(range(n))

    def find(self, x: int) -> int:
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x: int, y: int) -> None:
        self.par[self.find(x)] = self.find(y)
```

#### Template2

记录树的高度 + 路径压缩

```python
class UnionFind:
    def __init__(self, n: int):
        self.par = list(range(n))  # 祖先结点
        self.rank = [0] * n  # 树的高度

    def find(self, x: int) -> int:
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1
```

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

# 字符串

## KMP 算法

```python
from typing import List


# 计算失配指针
def cal_fails(p: str) -> List[int]:
    k, j, n = -1, 0, len(p)
    fails = [-1] * n
    while j < n - 1:
        if k == -1 or p[k] == p[j]:
            k += 1
            j += 1
            fails[j] = k
        else:
            k = fails[k]
    return fails


# 匹配
def search(s: str, p: str) -> int:
    fails = cal_fails(p)
    i, j, sl, pl = 0, 0, len(s), len(p)
    while i < sl and j < pl:
        if j == -1 or s[i] == p[j]:
            i += 1
            j += 1
        else:
            j = fails[j]
    return i - j if j == pl else -1
```

### 最长回文前缀

```python
"""
具体地，记 s' 为 s 的反序，由于 s1 是 s 的前缀，那么 s1' 就是 s' 的后缀。
考虑到 s1 是一个回文串，因此 s1 = s1'，s1 同样是 s' 的后缀。

这样一来，我们将 s 作为模式串，s' 作为查询串进行匹配。当遍历到 s' 的末尾时，
如果匹配到 s 中的第 i 个字符，那么说明 s 的前 i 个字符与 s' 的后 i 个字符
相匹配（即相同），s 的前 i 个字符对应 s1，s' 的后 i 个字符对应 s1'，由于
s1 = s1'，因此 s1 就是一个回文串。

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
```

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

## 排序网络
