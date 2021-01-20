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

```python
def gcd(x: int, y: int) -> int:
    return x if y == 0 else gcd(y, x % y)


if __name__ == "__main__":
    print(gcd(12, 4))
    print(gcd(12, 3))
```

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


if __name__ == "__main__":
    print(eratosthenes_sieve(100))
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


if __name__ == "__main__":
    print(euler_sieve(100))

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


if __name__ == "__main__":
    print(quick_pow(2, 10))
    print(quick_pow(2.0, 10))
    print(quick_pow(2.0, -10))
```

## 格雷码

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。格雷编码序列必须以 0 开头。

**方法一：**

格雷编码：

设 G(n) 表示总位数为 n 的各类编码集合，根据以下策略可以求出 G(n+1)

1. 将 G(n) 的每个元素前添加 0 得到 G'(n)；
2. 将 G'(n) 反转得到镜像 R(n)，在 R(n) 中的每个元素前添加 1 得到 R'(n)；
3. 将 G'(n) 与 R'(n) 合并得到 G(n+1)；

编码思路：

1. 初始化 G(0) 和位数标识 base；

2. 外层循环次数为总位数 n；

3. 内层循环倒序遍历 res，位数标识加上当前索引对应的值，即为 R'(n) 中的元素；
4. 在 res 后追加上述计算的元素，遍历结束，得到 Gray 编码集；

**方法二：**

二进制转格雷码公式：

```text
某二进制数：         B[n-1],B[n-2],...,B[2],B[1],B[0]
其对应的格雷码：      G[n-1],G[n-2],...,G[2],G[1],G[0]
其中：最高位保留      G[n-1] = B[n-1]
     其它各位        G[i] = B[i+1] ^ B[i],  i = 0, 1, 2, ..., n-2

https://pic.leetcode-cn.com/1013850d7f6c8cf1d99dc0ac3292264b74f6a52d84e0215f540c80952e184f41-image.png

例如：二进制数为      1  0  1  1  0
                   |\ |\ |\ |\ |  # noqa W605
                   | \| \| \| \|  # noqa W605
                   |  ^  ^  ^  ^
                   |  |  |  |  |
     格雷码为：      1  1  1  0  1
```

遍历 0 到 2^(n-1)，利用公式转换，即最高位保留，其它位是当前位和它的高一位进行异或操作。

```python
from typing import List


def binary2graycode(n: int) -> int:
    return n ^ (n >> 1)


def gen_gray_code_seq(n: int) -> List[int]:
    ans, base = [0], 1
    for i in range(n):
        for j in range(len(ans) - 1, -1, -1):
            ans.append(base + ans[j])
        base <<= 1
    return ans


def gen_gray_code_seq2(n: int) -> List[int]:
    ans = []
    start, end = 0, 1 << n
    while start < end:
        ans.append(start ^ (start >> 1))
        start += 1
    return ans


if __name__ == "__main__":
    print(gen_gray_code_seq(3))
    print(gen_gray_code_seq2(3))

    print(gen_gray_code_seq(4))
    print(gen_gray_code_seq2(4))

    print(binary2graycode(2))  # 2(10) => 3(11)
    print(binary2graycode(3))  # 3(11) => 2(10)
    print(binary2graycode(6))  # 6(110) => 5(101)
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


if __name__ == "__main__":
    print(binarySearch([-1, 0, 3, 5, 9, 12], 9))
    print(binarySearch([-1, 0, 3, 5, 9, 12], 2))
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


if __name__ == "__main__":
    print(binarySearch([1, 2, 3, 4, 5], 3))

    print('\n================= 搜索左边界 =================')
    print(searchLeftMargin([5, 7, 7, 8, 8, 10], 8))
    print(searchLeftMargin([5, 7, 7, 8, 8, 10], 6))
    print(searchLeftMargin([7, 7, 8, 8], 7))
    print(searchLeftMargin([7, 7, 8, 8], 8))
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


if __name__ == "__main__":
    print(binarySearch([1, 2, 3, 4, 5], 3))

    print('\n================= 搜索右边界 =================')
    print(searchRightMargin([5, 7, 7, 8, 8, 10], 8))
    print(searchRightMargin([5, 7, 7, 8, 8, 10], 6))
    print(searchRightMargin([7, 7, 8, 8], 7))
    print(searchRightMargin([7, 7, 8, 8], 8))
```

#### 模板 III

模板 #3 是二分查找的另一种独特形式。它用于搜索需要访问当前索引及其在数组中的直接左右邻居索引的元素或条件。

**关键属性**

- 实现二分查找的另一种方法。
- 搜索条件需要访问元素的直接左右邻居。
- 使用元素的邻居来确定它是向右还是向左。
- 保证查找空间在每个步骤中至少有 3 个元素。
- 需要进行后处理。当剩下 2 个元素时，循环 / 递归结束。需要评估其余元素是否符合条件。

**区分语法**

- 初始条件：left = 0, right = length-1
- 终止：left + 1 == right
- 向左查找：right = mid
- 向右查找：left = mid

```python
def binarySearch(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target:
        return left
    if nums[right] == target:
        return right
    return -1
```

### 数组

#### 前缀和

```python
from typing import List


def get_prefix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ps = [0] * (n + 1)
    for i in range(n):
        ps[i + 1] = ps[i] + nums[i]
    return ps


if __name__ == "__main__":
    print(get_prefix_sum([1, 2, 3, 4, 5]))
```

#### 后缀和

```python
from typing import List


def get_suffix_sum(nums: List[int]) -> List[int]:
    n = len(nums)
    ss = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        ss[i] = ss[i + 1] + nums[i]
    return ss


if __name__ == "__main__":
    print(get_suffix_sum([5, 4, 3, 2, 1]))
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


if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(transpose(matrix))
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
```

#### 快慢指针判断链表是否有环

```python
def hasCycle(self, head: ListNode) -> bool:
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

堆（Heap）是计算机科学中的一种特别的树状数据结构。若是满足以下特性，即可称为堆：“给定堆中任意节点 P 和 C，若 P 是 C 的父节点，那么 P的值会小于等于（或大于等于） C 的值”。

堆始于 J._W._J._Williams 在 1964 年发表的堆排序（heap sort），当时他提出了二叉堆树作为此算法的数据结构。堆在 Dijkstra 等几种有效的图形算法中也非常重要。

堆是一种称为优先级队列的抽象数据类型的最有效实现，实际上优先级队列通常被称为“堆”，而不管它们如何实现。

严格来说，堆也有不同的种类。二叉堆本质上是完全二叉树，可以分为两种类型：

- 最大堆（大顶堆）
- 最小堆（小顶堆）

二叉堆的根节点叫做 堆顶。最大堆和最小堆的特点，决定了在最大堆的堆顶是整个堆中的 最大元素；最小堆的堆顶是整个堆中的 最小元素。

**堆的逻辑结构与物理存储：**

二叉堆在逻辑上虽然是一颗完全二叉树，但它的存储方式并不是链式存储，而是顺序存储。换句话说，二叉堆的所有节点都存储在数组当中。

树的节点是按从上到下、从左到右的顺序紧凑排列的。

利用数组下标作为节点编号，假设父节点的下标是 parent，则有

- `左儿子的下标 = 2 * parent + 1`
- `右儿子的下标 = 2 * parent + 2`

```python
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
```

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

KMP 利用之前已经部分匹配这个有效信息，保持 i 不回溯，通过修改 j 的位置，让模式串尽量地移动到有效的位置，而模式串移动的信息存在 next 数组中。

KMP 算法的关键是 next 数组的计算，next 数组的计算只与模式串有关，next 中存储的值为当前下标之前的子串的 最长相同前缀和后缀的长度。

如模式串 "ABCDABD" 的 next 数组：

```text
    i         0  1  2  3  4  5  6
    pattern   A  B  C  D  A  B  D
    next[i]  -1	 0  0  0  0  1  2
```

注：next[0] 设为 -1

已知 `next[j] = k`，如何求出 `next[j+1]`？

- 如果 `p[j] == p[k]` ，则 `next[j+1] = next[j] + 1 = k + 1`；

- 如果 `p[j] != p[k]`， 则令 `k = next[k]`；

  - 如果此时 `p[j] == p[k]`，则 `next[j+1] = k+1`；
  - 如果不相等，则继续递归前缀索引，令 `k = next[k]`，继续判断，直至 `k = -1` (即 `k = next[0]` )​  或者 `p[j] == p[k]` 为止；

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


if __name__ == '__main__':
    pattern = 'ABCDABD'
    fails = cal_fails(pattern)
    print(fails)

    source = 'BBC ABCDAB ABCDABCDABDE'
    start = search(source, pattern)
    end = start + len(pattern)

    for ch in source:
        print('{: >2}'.format(ch), end=' ')
    print()
    for i in range(len(source)):
        print('{: >2}'.format(i), end=' ')
    print()

    print(start, end, source[start:end])
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

快速排序（Quicksort），又称分区交换排序（partition-exchange sort），简称快排，是一种高效的排序算法。由英国计算机科学家 Tony Hoare 于 1959年开发并于 1961 年发表，至今它仍然是一种常用的排序算法。事实上，如果实施得当，它可以比归并排序、堆排序快两到三倍。

**基本原理：**

快速排序是图灵奖得主 C. R. A. Hoare 于 1960 年提出的一种划分交换排序，它采用了一种分治的策略，通常称其为 分治法(Divide-and-ConquerMethod)。分治法的基本思想是：将原问题分解为若干个规模更小但结构与原问题相似的子问题。递归地求解这些子问题，然后将这些子问题的解组合为原问题的解。

**算法步骤：**

1. 从序列中挑出一个元素，作为"基准" (pivot)；
2. 把所有比基准值小的元素放在基准前面，所有比基准值大的元素放在基准值的后面（相同的数可以放到任一边），这个操作称为分区 (partition) 操作，分区操作结束后，基准元素所处的位置就是最终排序后它的位置；
3. 递归地把小于基准值元素的子数列和大于基准值元素的子数列排序，直到所有子序列的大小为 0 或 1，这时整体已经排好序了；

```python
from typing import List


def partition(nums: List[int], left: int, right: int) -> None:
    j = left - 1
    for i in range(left, right):
        if nums[i] <= nums[right]:
            j += 1
            nums[i], nums[j] = nums[j], nums[i]
    j += 1
    nums[j], nums[right] = nums[right], nums[j]
    return j


def quick_sort(nums: List[int], left: int, right: int) -> None:
    if left < right:
        idx = partition(nums, left, right)
        quick_sort(nums, left, idx - 1)
        quick_sort(nums, idx + 1, right)


if __name__ == "__main__":
    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    quick_sort(nums, 0, len(nums) - 1)
    print(nums)
```

## 堆排序

## 归并排序

归并排序（Merge sort），是创建在归并操作上的一种基于比较的排序算法。1945 年由约翰·冯·诺伊曼首次提出。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用，且各层分治递归可以同时进行。

**基本原理：**

归并排序算法是分治策略实现对 n 个元素进行排序的算法。其基本思想是：将待排序元素分成大小大致相同的 2 个子集合，分别对 2 个子集合进行排序，最终将排好序的子集合合并成为所要求的排好序的集合。

**算法步骤：**

1. 申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列；
2. 设定两个指针，最初位置分别为两个已经排序序列的起始位置；
3. 比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置；
4. 重复步骤 3 直到某一指针到达序列尾；
5. 将另一序列剩下的所有元素直接复制到合并序列尾；

```shell
from typing import List


def merge(nums: List[int], left: int, mid: int, right: int) -> None:
    tmp = [0] * (right - left + 1)
    i, j, k = left, mid + 1, 0
    while i <= mid or j <= right:
        if i > mid:
            tmp[k] = nums[j]
            j += 1
        elif j > right:
            tmp[k] = nums[i]
            i += 1
        else:
            if nums[i] <= nums[j]:
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
        k += 1

    for i in range(k):
        nums[left + i] = tmp[i]


def merge_sort(nums: List[int], left: int, right: int) -> None:
    if left < right:
        mid = (left + right) // 2
        merge_sort(nums, left, mid)
        merge_sort(nums, mid + 1, right)
        merge(nums, left, mid, right)


def merge_sort2(nums: List[int], left: int, right: int) -> None:
    size, n = 1, len(nums)
    while size < n:
        left = 0
        while left + size < n:
            mid = left + size - 1
            right = min(mid + size, n - 1)
            merge(nums, left, mid, right)
            left = right + 1
        size *= 2


if __name__ == "__main__":
    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    merge_sort(nums, 0, len(nums) - 1)
    print(nums)

    nums = [3, 1, 4, 9, 6, 0, 7, 2, 5, 8]
    merge_sort2(nums, 0, len(nums) - 1)
    print(nums)
```

## 基数排序

## 排序网络
