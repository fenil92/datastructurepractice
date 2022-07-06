# 1. Contains Duplicate 


Given an integer array  `nums`, return  `true`  if any value appears  **at least twice**  in the array, and return  `false`  if every element is distinct.

**Example 1:**

**Input:** nums = [1,2,3,1]
**Output:** true

**Example 2:**

**Input:** nums = [1,2,3,4]
**Output:** false

**Example 3:**

**Input:** nums = [1,1,1,3,3,4,3,2,4,2]
**Output:** true

**Constraints:**

-   `1 <= nums.length <= 105`
-   `-109  <= nums[i] <= 109`

`Arrays LTE 217`
``` python	
        class Solution:
	        def containsDuplicate(self, nums: List[int]) -> bool:
                lookup = {}
                
                for i in range(len(nums)):
                    if nums[i] in lookup:
                        return True
                    else:
                        lookup[nums[i]] = 1
                        
                return False
```
---
# 2. Missing Number

Given an array  `nums`  containing  `n`  distinct numbers in the range  `[0, n]`, return  _the only number in the range that is missing from the array._

**Example 1:**

**Input:** nums = [3,0,1]
**Output:** 2
**Explanation:** n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.

**Example 2:**

**Input:** nums = [0,1]
**Output:** 2
**Explanation:** n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.

**Example 3:**

**Input:** nums = [9,6,4,2,3,5,7,0,1]
**Output:** 8
**Explanation:** n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.

**Constraints:**

-   `n == nums.length`
-   `1 <= n <= 104`
-   `0 <= nums[i] <= n`
-   All the numbers of  `nums`  are  **unique**.

**Follow up:**  Could you implement a solution using only  `O(1)`  extra space complexity and  `O(n)`  runtime complexity?

`Arrays Bit Manipulation LTE 268`
``` python
    class Solution:
        def missingNumber(self, nums: List[int]) -> int:
            lookup = set()
            
            for n in nums:
                lookup.add(n)
                
            for i in range(len(nums)+1):
                if not i in lookup:
                    return i
                
            return None
```
---

# 3. Find All Numbers Disappeared in an Array.



Given an array  `nums`  of  `n`  integers where  `nums[i]`  is in the range  `[1, n]`, return  _an array of all the integers in the range_  `[1, n]`  _that do not appear in_  `nums`.

**Example 1:**

**Input:** nums = [4,3,2,7,8,2,3,1]
**Output:** [5,6]

**Example 2:**

**Input:** nums = [1,1]
**Output:** [2]

**Constraints:**

-   `n == nums.length`
-   `1 <= n <= 105`
-   `1 <= nums[i] <= n`

**Follow up:**  Could you do it without extra space and in  `O(n)`  runtime? You may assume the returned list does not count as extra space.

`Arrays LTE 448`

``` python
    class Solution:
        def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
            missing = []
            lookup = set()
            
            for n in nums:
                lookup.add(n)
                
            for i in range(1, len(nums)+1):
                if not i in lookup:
                    missing.append(i)
                    
            return missing
```
---
# 4. Single Number

Given a  **non-empty** array of integers  `nums`, every element appears  _twice_  except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example 1:**

**Input:** nums = [2,2,1]
**Output:** 1

**Example 2:**

**Input:** nums = [4,1,2,1,2]
**Output:** 4

**Example 3:**

**Input:** nums = [1]
**Output:** 1

**Constraints:**

-   `1 <= nums.length <= 3 * 104`
-   `-3 * 104  <= nums[i] <= 3 * 104`
-   Each element in the array appears twice except for one element which appears only once.

`Array LTE 136`
``` python
    class Solution:
        def singleNumber(self, nums: List[int]) -> int:
            nums.sort()
        
            left, right = 0 , 1
            
            while right < len(nums):
                print(nums[left], nums[right])
                if nums[left] != nums[right]:
                    return nums[left]
                else:
                    left +=2
                    right+=2
                    
            return nums[left]
```
---
# 5. Climbing Stairs

You are climbing a staircase. It takes  `n`  steps to reach the top.

Each time you can either climb  `1`  or  `2`  steps. In how many distinct ways can you climb to the top?

**Example 1:**

**Input:** n = 2
**Output:** 2
**Explanation:** There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

**Example 2:**

**Input:** n = 3
**Output:** 3
**Explanation:** There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

**Constraints:**

-   `1 <= n <= 45`

`Dynamic Programming LTE 70`
``` python
    class Solution:
        def climbStairs(self, n: int) -> int:
        
            one, two = 1, 1
            
            # dynamic programming bottom up approach
            for i in range(1, n):
                temp = one
                one = one + two
                two = temp
                
            return one
```
---
# 6. Best Time to Buy and Sell Stock

You are given an array  `prices`  where  `prices[i]`  is the price of a given stock on the  `ith`  day.

You want to maximize your profit by choosing a  **single day**  to buy one stock and choosing a  **different day in the future**  to sell that stock.

Return  _the maximum profit you can achieve from this transaction_. If you cannot achieve any profit, return  `0`.

**Example 1:**

**Input:** prices = [7,1,5,3,6,4]
**Output:** 5
**Explanation:** Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

**Example 2:**

**Input:** prices = [7,6,4,3,1]
**Output:** 0
**Explanation:** In this case, no transactions are done and the max profit = 0.

**Constraints:**

-   `1 <= prices.length <= 105`
-   `0 <= prices[i] <= 104`

`Greedy LTE 121`
``` python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            result = 0
            l, r = 0 , 1
            while( l < r and r < len(prices)):
                if prices[r] < prices[l]:
                    l = r
                elif(prices[r] - prices[l] > result):
                    result = prices[r] - prices[l]
                    
                r+=1
            
            return result
```
---
# 7. Maximum Subarray

Given an integer array  `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return  _its sum_.

A  **subarray**  is a  **contiguous**  part of an array.

**Example 1:**

**Input:** nums = [-2,1,-3,4,-1,2,1,-5,4]
**Output:** 6
**Explanation:** [4,-1,2,1] has the largest sum = 6.

**Example 2:**

**Input:** nums = [1]
**Output:** 1

**Example 3:**

**Input:** nums = [5,4,-1,7,8]
**Output:** 23

**Constraints:**

-   `1 <= nums.length <= 105`
-   `-104  <= nums[i] <= 104`

**Follow up:**  If you have figured out the  `O(n)`  solution, try coding another solution using the  **divide and conquer**  approach, which is more subtle.

`Dynamic Programming LTM 53`
``` python
    class Solution:
        def maxSubArray(self, nums: List[int]) -> int:
        
            maxSum = nums[0]
            curSum = 0
            
            for n in nums:
                if curSum < 0:
                    curSum = 0

                curSum += n
                maxSum = max(curSum, maxSum) 
            
            return maxSum
```
---
# 8. Range Sum Query - Immutable
Given an integer array  `nums`, handle multiple queries of the following type:

1.  Calculate the  **sum**  of the elements of  `nums`  between indices  `left`  and  `right`  **inclusive**  where  `left <= right`.

Implement the  `NumArray`  class:

-   `NumArray(int[] nums)`  Initializes the object with the integer array  `nums`.
-   `int sumRange(int left, int right)`  Returns the  **sum**  of the elements of  `nums`  between indices  `left`  and  `right`  **inclusive**  (i.e.  `nums[left] + nums[left + 1] + ... + nums[right]`).

**Example 1:**

**Input**
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
**Output**
[null, 1, -1, -3]

**Explanation**
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3

**Constraints:**

-   `1 <= nums.length <= 104`
-   `-105  <= nums[i] <= 105`
-   `0 <= left <= right < nums.length`
-   At most  `104`  calls will be made to  `sumRange`.

`Dynamic Programming LTE 303`
``` python
    class NumArray:

        def __init__(self, nums: List[int]):
            self.cummulative = [0]
            for n in nums:
                self.cummulative.append(self.cummulative[-1]+n)
            

        def sumRange(self, left: int, right: int) -> int:
            return self.cummulative[right+1] - self.cummulative[left]
                    
        # Your NumArray object will be instantiated and called as such:
        # obj = NumArray(nums)
        # param_1 = obj.sumRange(left,right)
```
---
# 9. Counting Bits

Given an integer  `n`, return  _an array_ `ans` _of length_ `n + 1` _such that for each_ `i`  (`0 <= i <= n`)_,_ `ans[i]` _is the  **number of**_ `1`_**'s**  in the binary representation of_ `i`.

**Example 1:**

**Input:** n = 2
**Output:** [0,1,1]
**Explanation:**
0 --> 0
1 --> 1
2 --> 10

**Example 2:**

**Input:** n = 5
**Output:** [0,1,1,2,1,2]
**Explanation:**
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101

**Constraints:**

-   `0 <= n <= 105`

**Follow up:**

-   It is very easy to come up with a solution with a runtime of  `O(n log n)`. Can you do it in linear time  `O(n)`  and possibly in a single pass?
-   Can you do it without using any built-in function (i.e., like  `__builtin_popcount`  in C++)?

`Dynamic Programming LTE 338`
``` python
    class Solution:
        def countBits(self, n: int) -> List[int]:
            dp = [0] * (n+1)
            offset = 1
            # dynamic programming approach
            
            for i in range(1, n+1):
                if offset*2 == i:
                    offset = i
                dp[i] = 1 + dp[i-offset]
                
            return dp
```
---
# 10. Linked List Cycle

Given  `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally,  `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.

Return `true` _if there is a cycle in the linked list_. Otherwise, return  `false`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

**Input:** head = [3,2,0,-4], pos = 1
**Output:** true
**Explanation:** There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

**Example 2:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)

**Input:** head = [1,2], pos = 0
**Output:** true
**Explanation:** There is a cycle in the linked list, where the tail connects to the 0th node.

**Example 3:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)

**Input:** head = [1], pos = -1
**Output:** false
**Explanation:** There is no cycle in the linked list.

**Constraints:**

-   The number of the nodes in the list is in the range  `[0, 104]`.
-   `-105  <= Node.val <= 105`
-   `pos`  is  `-1`  or a  **valid index**  in the linked-list.

**Follow up:**  Can you solve it using  `O(1)`  (i.e. constant) memory?

`Fast and slow pointers LTE 141`      
```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution:
        def hasCycle(self, head: Optional[ListNode]) -> bool:
            slow, fast = head, head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    return True
            
            return False
```
---
# 11. Middle of the Linked List

Given the  `head`  of a singly linked list, return  _the middle node of the linked list_.

If there are two middle nodes, return  **the second middle**  node.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist1.jpg)

**Input:** head = [1,2,3,4,5]
**Output:** [3,4,5]
**Explanation:** The middle node of the list is node 3.

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist2.jpg)

**Input:** head = [1,2,3,4,5,6]
**Output:** [4,5,6]
**Explanation:** Since the list has two middle nodes with values 3 and 4, we return the second one.

**Constraints:**

-   The number of nodes in the list is in the range  `[1, 100]`.
-   `1 <= Node.val <= 100`

`Fast and slow pointers LTE 876`
``` python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    
    class Solution:
        def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
            slow, fast = head, head
            # when fast has reached end slow will be in middle
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
            return slow
```
---
# 12. Palindrome Linked List

Given the  `head`  of a singly linked list, return  `true`  if it is a palindrome.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

**Input:** head = [1,2,2,1]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg)

**Input:** head = [1,2]
**Output:** false

**Constraints:**

-   The number of nodes in the list is in the range  `[1, 105]`.
-   `0 <= Node.val <= 9`

**Follow up:** Could you do it in `O(n)` time and `O(1)` space?

`Fast and slow pointers LTE 234`
``` python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def isPalindrome(self, head: Optional[ListNode]) -> bool:
            
            slow, fast = head, head
            
            # find the middle
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next
                
            # reverse the second half
            prev = None
            while slow:
                nxt = slow.next
                slow.next = prev
                prev = slow
                slow = nxt
            
            # check for palindrome
            left, right = head, prev
            while left and right:
                if left.val != right.val:
                    return False
                left = left.next
                right = right.next
            
            # Another way to solve this problem is to iterate through linked list and create an array, then check for array palindrome. But it will require O(n) space.
            return True
```
---
# 13. Remove Linked List Elements

Given the  `head`  of a linked list and an integer  `val`, remove all the nodes of the linked list that has  `Node.val == val`, and return  _the new head_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

**Input:** head = [1,2,6,3,4,5,6], val = 6
**Output:** [1,2,3,4,5]

**Example 2:**

**Input:** head = [], val = 1
**Output:** []

**Example 3:**

**Input:** head = [7,7,7,7], val = 7
**Output:** []

**Constraints:**

-   The number of nodes in the list is in the range  `[0, 104]`.
-   `1 <= Node.val <= 50`
-   `0 <= val <= 50`

`Fast and slow pointers LTE 203`
```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
            dummy = ListNode(next=head)
            cur = head
            prev = dummy
            
            while cur:
                nxt = cur.next
                if cur.val == val:
                    prev.next = nxt
                else:
                    prev = cur
                    
                cur = nxt
                
            return dummy.next
 ```
 ---
                
