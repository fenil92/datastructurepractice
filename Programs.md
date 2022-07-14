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
# 14. Remove Duplicates from Sorted List


Given the  `head`  of a sorted linked list,  _delete all duplicates such that each element appears only once_. Return  _the linked list  **sorted**  as well_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)

**Input:** head = [1,1,2]
**Output:** [1,2]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/01/04/list2.jpg)

**Input:** head = [1,1,2,3,3]
**Output:** [1,2,3]

**Constraints:**

-   The number of nodes in the list is in the range  `[0, 300]`.
-   `-100 <= Node.val <= 100`
-   The list is guaranteed to be  **sorted**  in ascending order.
`Fast and slow pointers LTE 83`
```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode(val="dummy", next=head)
            prev = dummy
            cur = head
            
            while cur:
                nxt = cur.next
                if prev.val == cur.val:
                    prev.next = nxt
                else:
                    prev = cur
                    
                cur = nxt
                
            return dummy.next
```
---
# 15. Reverse Linked List


Given the  `head`  of a singly linked list, reverse the list, and return  _the reversed list_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

**Input:** head = [1,2,3,4,5]
**Output:** [5,4,3,2,1]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)

**Input:** head = [1,2]
**Output:** [2,1]

**Example 3:**

**Input:** head = []
**Output:** []

**Constraints:**

-   The number of nodes in the list is the range  `[0, 5000]`.
-   `-5000 <= Node.val <= 5000`

**Follow up:**  A linked list can be reversed either iteratively or recursively. Could you implement both?
`Fast and slow pointers LTE 206` 
```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            prev = None
            cur = head
            
            while cur:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
                
            return prev
```
---
# 16. Merge Two Sorted Lists

You are given the heads of two sorted linked lists  `list1`  and  `list2`.

Merge the two lists in a one  **sorted**  list. The list should be made by splicing together the nodes of the first two lists.

Return  _the head of the merged linked list_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

**Input:** list1 = [1,2,4], list2 = [1,3,4]
**Output:** [1,1,2,3,4,4]

**Example 2:**

**Input:** list1 = [], list2 = []
**Output:** []

**Example 3:**

**Input:** list1 = [], list2 = [0]
**Output:** [0]

**Constraints:**

-   The number of nodes in both lists is in the range  `[0, 50]`.
-   `-100 <= Node.val <= 100`
-   Both  `list1`  and  `list2`  are sorted in  **non-decreasing**  order.
`Two pointers LTE 21`
```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
                dummy = ListNode()
                cur = dummy
                
                while(list1 and list2):
                    if(list1.val < list2.val):
                        cur.next = list1
                        list1 = list1.next
                    else:
                        cur.next = list2
                        list2 = list2.next
                    cur = cur.next
                    
                if list1:
                    cur.next = list1
                    
                if list2:
                    cur.next = list2
                    
                return dummy.next
```
---
# 17. Binary Search    


Given an array of integers  `nums`  which is sorted in ascending order, and an integer  `target`, write a function to search  `target`  in  `nums`. If  `target`  exists, then return its index. Otherwise, return  `-1`.

You must write an algorithm with  `O(log n)`  runtime complexity.

**Example 1:**

**Input:** nums = [-1,0,3,5,9,12], target = 9
**Output:** 4
**Explanation:** 9 exists in nums and its index is 4

**Example 2:**

**Input:** nums = [-1,0,3,5,9,12], target = 2
**Output:** -1
**Explanation:** 2 does not exist in nums so return -1

**Constraints:**

-   `1 <= nums.length <= 104`
-   `-104  < nums[i], target < 104`
-   All the integers in  `nums`  are  **unique**.
-   `nums`  is sorted in ascending order.

`Binary search LTE 704`   
```python
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            l, r = 0, len(nums)-1
            m = (l+r)//2
        
            while(l <= r):
                if target == nums[m]:
                    return m
                elif target < nums[m]:
                    r = m-1
                else:
                    l = m+1          
                m = (l+r)//2
                
            return -1
```
---
# 18. Find Smallest Letter Greater Than Target

Given a characters array  `letters`  that is sorted in  **non-decreasing**  order and a character  `target`, return  _the smallest character in the array that is larger than_ `target`.

**Note**  that the letters wrap around.

-   For example, if  `target == 'z'`  and  `letters == ['a', 'b']`, the answer is  `'a'`.

**Example 1:**

**Input:** letters = ["c","f","j"], target = "a"
**Output:** "c"

**Example 2:**

**Input:** letters = ["c","f","j"], target = "c"
**Output:** "f"

**Example 3:**

**Input:** letters = ["c","f","j"], target = "d"
**Output:** "f"

**Constraints:**

-   `2 <= letters.length <= 104`
-   `letters[i]`  is a lowercase English letter.
-   `letters`  is sorted in  **non-decreasing**  order.
-   `letters`  contains at least two different characters.
-   `target`  is a lowercase English letter.

`Binary search LTE 744`
```python
        class Solution:
            def nextGreatestLetter(self, letters: List[str], target: str) -> str:
                # binary search
                n = len(letters)
                l, r = 0, n
                if letters[n-1].upper() <= target.upper() or target.upper() < letters[0].upper():
                    return letters[0]
                
                m = (l+r)//2
                
                while(l <= r):
                    if target.upper() >= letters[m].upper():
                        l = m+1
                    else:
                        r = m-1
                    m = (l+r)//2
                
            
                return letters[m+1]
```
---
# 19. Peak Index in a Mountain Array

Let's call an array  `arr`  a  **mountain** if the following properties hold:

-   `arr.length >= 3`
-   There exists some  `i`  with `0 < i < arr.length - 1` such that:
    -   `arr[0] < arr[1] < ... arr[i-1] < arr[i]`
    -   `arr[i] > arr[i+1] > ... > arr[arr.length - 1]`

Given an integer array  `arr`  that is  **guaranteed**  to be a mountain, return any `i` such that `arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1]`.

**Example 1:**

**Input:** arr = [0,1,0]
**Output:** 1

**Example 2:**

**Input:** arr = [0,2,1,0]
**Output:** 1

**Example 3:**

**Input:** arr = [0,10,5,2]
**Output:** 1

**Constraints:**

-   `3 <= arr.length <= 104`
-   `0 <= arr[i] <= 106`
-   `arr`  is  **guaranteed**  to be a mountain array.

**Follow up:** Finding the `O(n)` is straightforward, could you find an `O(log(n))` solution?
`Binary search LTE 852`

```python
    class Solution:
        def peakIndexInMountainArray(self, arr: List[int]) -> int:
            left, right = 0 , len(arr)
            
            while(left <= right):
                m = (left+right)//2
                
                if(arr[m] < arr[m+1]):
                    left = m+1
                else:
                    right=m-1
                    
            return left
```
---
# 20. Average of Levels in Binary Tree


Given the `root` of a binary tree, return _the average value of the nodes on each level in the form of an array_. Answers within `10-5` of the actual answer will be accepted.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/09/avg1-tree.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/09/avg2-tree.jpg)

**Input:** root = [3,9,20,15,7]
**Output:** [3.00000,14.50000,11.00000]

**Constraints:**

-   The number of nodes in the tree is in the range  `[1, 104]`.
-   `-231  <= Node.val <= 231  - 1`

`BFS LTE 637`
```python
        # Definition for a binary tree node.
        # class TreeNode:
        #     def __init__(self, val=0, left=None, right=None):
        #         self.val = val
        #         self.left = left
        #         self.right = right
        class Solution:
            def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
                q = collections.deque()
                avg = []
                res = []
                q.append(root)
                
                while len(q) > 0:
                    qLen = len(q)
                    level = []
                    for i in range(qLen):
                        node = q.popleft()
                        if node: 
                            level.append(node.val)
                            if node.left:
                                q.append(node.left)
                            if node.right:
                                q.append(node.right)
                    if level:
                        res.append(level)
                
                for i in range(len(res)):
                    avg.append(sum(res[i])/len(res[i]))
                
                return avg
```
---                
 # 21. Minimum Depth of Binary Tree
 
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

**Note:** A leaf is a node with no children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** 2

**Example 2:**

**Input:** root = [2,null,3,null,4,null,5,null,6]
**Output:** 5

**Constraints:**

-   The number of nodes in the tree is in the range  `[0, 105]`.
-   `-1000 <= Node.val <= 1000`

`BFS DFS LTE 111`

```python
        # Definition for a binary tree node.
        # class TreeNode:
        #     def __init__(self, val=0, left=None, right=None):
        #         self.val = val
        #         self.left = left
        #         self.right = right
        class Solution:
            def minDepth(self, root: Optional[TreeNode]) -> int:
                if not root:
                    return 0
                if not root.left and not root.right:
                    return 1
                if not root.left:
                    return 1+ self.minDepth(root.right)
                if not root.right:
                    return 1+ self.minDepth(root.left)
                
                #recursive depth first search
                return 1+ min(self.minDepth(root.left), self.minDepth(root.right))
```            
---
# 22. Same Tree

Given the roots of two binary trees  `p`  and  `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)

**Input:** p = [1,2,3], q = [1,2,3]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/12/20/ex2.jpg)

**Input:** p = [1,2], q = [1,null,2]
**Output:** false

**Example 3:**

![](https://assets.leetcode.com/uploads/2020/12/20/ex3.jpg)

**Input:** p = [1,2,1], q = [1,1,2]
**Output:** false

**Constraints:**

-   The number of nodes in both trees is in the range  `[0, 100]`.
-   `-104  <= Node.val <= 104`

`DFS LTE 100`

```python
        # Definition for a binary tree node.
        # class TreeNode:
        #     def __init__(self, val=0, left=None, right=None):
        #         self.val = val
        #         self.left = left
        #         self.right = right
        class Solution:
            def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
                if not p and not q:
                    return True
                if not p or not q or p.val!=q.val:
                    return False
                
                return (self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right))
```
---
# 23.  Path Sum

Given the  `root`  of a binary tree and an integer  `targetSum`, return  `true`  if the tree has a  **root-to-leaf**  path such that adding up all the values along the path equals  `targetSum`.

A  **leaf**  is a node with no children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

**Input:** root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
**Output:** true
**Explanation:** The root-to-leaf path with the target sum is shown.

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

**Input:** root = [1,2,3], targetSum = 5
**Output:** false
**Explanation:** There two root-to-leaf paths in the tree:
(1 --> 2): The sum is 3.
(1 --> 3): The sum is 4.
There is no root-to-leaf path with sum = 5.

**Example 3:**

**Input:** root = [], targetSum = 0
**Output:** false
**Explanation:** Since the tree is empty, there are no root-to-leaf paths.

**Constraints:**

-   The number of nodes in the tree is in the range  `[0, 5000]`.
-   `-1000 <= Node.val <= 1000`
-   `-1000 <= targetSum <= 1000`

`DFS LTE 112`

```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
            
            def dfs(node, curSum):       
                if not node:
                    return False
                
                curSum += node.val
                if (not node.left and not node.right):
                    return curSum == targetSum
                
                return (dfs(node.left, curSum) or dfs(node.right, curSum))
            
            return dfs(root, 0)
```
---
# 24. Maximum Depth of Binary Tree
Given the  `root`  of a binary tree, return  _its maximum depth_.

A binary tree's  **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** 3

**Example 2:**

**Input:** root = [1,null,2]
**Output:** 2

**Constraints:**

-   The number of nodes in the tree is in the range  `[0, 104]`.
-   `-100 <= Node.val <= 100`

`DFS LTE 104`

```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def maxDepth(self, root: Optional[TreeNode]) -> int:
            if not root:
                return 0
            
            return 1+ max(self.maxDepth(root.left), self.maxDepth(root.right))
```
---
# 25. Diameter of Binary Tree

Given the  `root`  of a binary tree, return  _the length of the  **diameter**  of the tree_.

The  **diameter**  of a binary tree is the  **length**  of the longest path between any two nodes in a tree. This path may or may not pass through the  `root`.

The  **length**  of a path between two nodes is represented by the number of edges between them.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

**Input:** root = [1,2,3,4,5]
**Output:** 3
**Explanation:** 3 is the length of the path [4,2,1,3] or [5,2,1,3].

**Example 2:**

**Input:** root = [1,2]
**Output:** 1

**Constraints:**

-   The number of nodes in the tree is in the range  `[1, 104]`.
-   `-100 <= Node.val <= 100`

`DFS LTE 543`
```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
            res = [0]
            
            def dfs(node):
                
                if not node:
                    return -1 #height of null tree  
                
                left = dfs(node.left) #left height
                right = dfs(node.right) #right height
                res[0] = max(res[0], left +right+2)
                
                return 1+ max(left, right)
            
            dfs(root)
            return res[0]
```
---
# 26. Merge Two Binary Trees

You are given two binary trees  `root1`  and  `root2`.

Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

Return  _the merged tree_.

**Note:**  The merging process must start from the root nodes of both trees.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/05/merge.jpg)

**Input:** root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
**Output:** [3,4,5,5,4,null,7]

**Example 2:**

**Input:** root1 = [1], root2 = [1,2]
**Output:** [2,2]

**Constraints:**

-   The number of nodes in both trees is in the range  `[0, 2000]`.
-   `-104  <= Node.val <= 104`

`DFS LTE 617`
```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
            
            if not root1 and not root2:
                return None
            
            v1 = root1.val if root1 else 0
            v2 = root2.val if root2 else 0
            
            root = TreeNode(v1+v2)
            root.left = self.mergeTrees(root1.left if root1 else None, root2.left if root2 else None)
            root.right = self.mergeTrees(root1.right if root1 else None, root2.right if root2 else None)
            
            return root
```
---
# 27. Lowest Common Ancestor of a Binary Search Tree

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the  [definition of LCA on Wikipedia](https://en.wikipedia.org/wiki/Lowest_common_ancestor): “The lowest common ancestor is defined between two nodes  `p`  and  `q`  as the lowest node in  `T`  that has both  `p`  and  `q`  as descendants (where we allow  **a node to be a descendant of itself**).”

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

**Input:** root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
**Output:** 6
**Explanation:** The LCA of nodes 2 and 8 is 6.

**Example 2:**

![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

**Input:** root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
**Output:** 2
**Explanation:** The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

**Example 3:**

**Input:** root = [2,1], p = 2, q = 1
**Output:** 2

**Constraints:**

-   The number of nodes in the tree is in the range  `[2, 105]`.
-   `-109  <= Node.val <= 109`
-   All  `Node.val`  are  **unique**.
-   `p != q`
-   `p`  and  `q`  will exist in the BST.

`DFS LTE 235`

```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution:
        def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            # check for empty tree
            if not root:
                return
            
            cur = root
            while cur:
                
                # check if p and q are greater than current node, means descendant will be on right subtree
                if p.val > cur.val and q.val > cur.val:
                    cur = cur.right
                elif p.val < cur.val and q.val < cur.val:
                    cur = cur.left
                else:
                    return cur
            
```
---
# 28. Subtree of Another Tree

Given the roots of two binary trees  `root`  and  `subRoot`, return  `true`  if there is a subtree of  `root`  with the same structure and node values of `subRoot`  and  `false`  otherwise.

A subtree of a binary tree  `tree`  is a tree that consists of a node in  `tree`  and all of this node's descendants. The tree  `tree`  could also be considered as a subtree of itself.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/04/28/subtree1-tree.jpg)

**Input:** root = [3,4,5,1,2], subRoot = [4,1,2]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/04/28/subtree2-tree.jpg)

**Input:** root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
**Output:** false

**Constraints:**

-   The number of nodes in the  `root`  tree is in the range  `[1, 2000]`.
-   The number of nodes in the  `subRoot`  tree is in the range  `[1, 1000]`.
-   `-104  <= root.val <= 104`
-   `-104  <= subRoot.val <= 104`

`DFS LTE 572`

```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
            
            #if the subRoot is empty, then return true
            if not subRoot:
                return True
            
            # if the root is null, then return False
            if not root:
                return False
            
            if self.isSameTree(root, subRoot):
                return True
                
                
            # If the root subtree does not match traverse through left and right subtree
            return (self.isSubtree(root.left, subRoot) or
                        self.isSubtree(root.right, subRoot))
                
        
        def isSameTree(self, root, subRoot):
            #reached the end of the tree
            if not root and not subRoot:
                return True
            
            # If root value of both tree matches
            if root and subRoot and root.val == subRoot.val:
                return (self.isSameTree(root.left, subRoot.left)
                        and self.isSameTree(root.right, subRoot.right))
            
            return False
        
```
---
# 29. Invert Binary Tree
Given the  `root`  of a binary tree, invert the tree, and return  _its root_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)

**Input:** root = [4,2,7,1,3,6,9]
**Output:** [4,7,2,9,6,3,1]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/14/invert2-tree.jpg)

**Input:** root = [2,1,3]
**Output:** [2,3,1]

**Example 3:**

**Input:** root = []
**Output:** []

**Constraints:**

-   The number of nodes in the tree is in the range  `[0, 100]`.
-   `-100 <= Node.val <= 100`

`DFS LTE 226`

```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            if not root:
                return None
                    
            # DFS recursive method
            if root.left and root.right:
                # swap nodes
                tmp = root.left
                root.left = root.right
                root.right = tmp
                    
            self.invertTree(root.left)
            self.invertTree(root.right)
            return root
```
---
# 30. Two Sum
Given an array of integers  `nums` and an integer  `target`, return  _indices of the two numbers such that they add up to  `target`_.

You may assume that each input would have  **_exactly_  one solution**, and you may not use the  _same_  element twice.

You can return the answer in any order.

**Example 1:**

**Input:** nums = [2,7,11,15], target = 9
**Output:** [0,1]
**Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1].

**Example 2:**

**Input:** nums = [3,2,4], target = 6
**Output:** [1,2]

**Example 3:**

**Input:** nums = [3,3], target = 6
**Output:** [0,1]

**Constraints:**

-   `2 <= nums.length <= 104`
-   `-109  <= nums[i] <= 109`
-   `-109  <= target <= 109`
-   **Only one valid answer exists.**

**Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?

`Two pointers LTE 1`

```python
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    if(nums[i]+nums[j] == target):
                        return [i, j]
                    
            return
            
```
---
# 31. Squares of a Sorted Array

Given an integer array  `nums`  sorted in  **non-decreasing**  order, return  _an array of  **the squares of each number**  sorted in non-decreasing order_.

**Example 1:**

**Input:** nums = [-4,-1,0,3,10]
**Output:** [0,1,9,16,100]
**Explanation:** After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].

**Example 2:**

**Input:** nums = [-7,-3,2,3,11]
**Output:** [4,9,9,49,121]

**Constraints:**

-   `1 <= nums.length <= 104`
-   `-104  <= nums[i] <= 104`
-   `nums`  is sorted in  **non-decreasing**  order.

**Follow up:** Squaring each element and sorting the new array is very trivial, could you find an `O(n)` solution using a different approach?

`LTE 977`

```python
    class Solution:
        def sortedSquares(self, nums: List[int]) -> List[int]:
            if not nums:
                return None
            
            result = []
            l , r = 0, len(nums)-1
            # two pointer
            while l <= r:
                if abs(nums[l]) <= abs(nums[r]):
                    result.append(nums[r]* nums[r])
                    # right is greater than left
                    r-=1
                else:
                    result.append(nums[l]* nums[l]) 
                    l+=1
            
            return result[::-1] #reverse
```
---
# 32. Backspace String Compare
Given two strings  `s`  and  `t`, return  `true`  _if they are equal when both are typed into empty text editors_.  `'#'`  means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

**Example 1:**

**Input:** s = "ab#c", t = "ad#c"
**Output:** true
**Explanation:** Both s and t become "ac".

**Example 2:**

**Input:** s = "ab##", t = "c#d#"
**Output:** true
**Explanation:** Both s and t become "".

**Example 3:**

**Input:** s = "a#c", t = "b"
**Output:** false
**Explanation:** s becomes "c" while t becomes "b".

**Constraints:**

-   `1 <= s.length, t.length <= 200`
-   `s`  and  `t`  only contain lowercase letters and  `'#'`  characters.

**Follow up:**  Can you solve it in  `O(n)`  time and  `O(1)`  space?

`LTE 844`

```python
    class Solution:
        def backspaceCompare(self, s: str, t: str) -> bool:
            l, r = len(s) -1, len(t) -1
            
            while l >= 0 or r >= 0:
                
                left_valid = get_next_valid(s, l)
                right_valid = get_next_valid(t, r)
                #print (left_valid, right_valid)
                # both pointers reached to the beginning of string which means all characters matched 
                if left_valid < 0 and right_valid < 0: 
                    return True
                
                # If one of the pointer reaches the beginning of string
                if left_valid < 0 or right_valid < 0:
                    return False
                
                # If characters does not match
                if s[left_valid] != t[right_valid]:
                    return False
                
                l = left_valid-1
                r = right_valid-1
            return True   

    def get_next_valid(s, index):
            backspace = 0
            
            while index>=0:
                if s[index] == "#":
                    backspace += 1
                elif backspace > 0:
                    backspace -=1
                else:
                    break
                index -=1 
            
            return index
        
                
        
```
---
# 33. Majority Element
Given an array  `nums`  of size  `n`, return  _the majority element_.

The majority element is the element that appears more than  `⌊n / 2⌋`  times. You may assume that the majority element always exists in the array.

**Example 1:**

**Input:** nums = [3,2,3]
**Output:** 3

**Example 2:**

**Input:** nums = [2,2,1,1,1,2,2]
**Output:** 2

**Constraints:**

-   `n == nums.length`
-   `1 <= n <= 5 * 104`
-   `-109  <= nums[i] <= 109`

**Follow-up:** Could you solve the problem in linear time and in `O(1)` space?

```python
    class Solution:
        def majorityElement(self, nums: List[int]) -> int:
            # To solve it in O(1) space, we require Boyer Moores algorithm
            res, maxCount = 0, 0
            for n in nums:
                if maxCount == 0:
                    res = n
                    maxCount = 1
                elif res == n:
                    maxCount +=1
                else:
                    maxCount -=1
            
            return res
                    
        
        
        '''   
        # frequency counter method is one of the way to solve this problem, however it takes O(n) space.
            # dictionary to store element frequency
        
            count= {}
            res, maxCount = 0, 0
            for n in nums:
                count[n] = 1 + count.get(n, 0)
                res = n if count[n] > maxCount else res
                maxCount = max(count[res], maxCount)
                
            return res
        '''
```
---
# 34.  Product of Array Except Self

Given an integer array  `nums`, return  _an array_  `answer`  _such that_  `answer[i]`  _is equal to the product of all the elements of_  `nums`  _except_  `nums[i]`.

The product of any prefix or suffix of  `nums`  is  **guaranteed**  to fit in a  **32-bit**  integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

**Example 1:**

**Input:** nums = [1,2,3,4]
**Output:** [24,12,8,6]

**Example 2:**

**Input:** nums = [-1,1,0,-3,3]
**Output:** [0,0,9,0,0]

**Constraints:**

-   `2 <= nums.length <= 105`
-   `-30 <= nums[i] <= 30`
-   The product of any prefix or suffix of  `nums`  is  **guaranteed**  to fit in a  **32-bit**  integer.

**Follow up:** Can you solve the problem in  `O(1)` extra space complexity? (The output array  **does not**  count as extra space for space complexity analysis.)

```python
    class Solution:
        def productExceptSelf(self, nums: List[int]) -> List[int]:
            '''
    main problem here is it is mentioned to not use division operator and solve it in O(n). If division operation was permissible, this would have been a very problem we can get the total product of array and then divide each element to get the resultant array.
            '''
            '''
            Another way to solve the problem is by traversing the array 2 times and saving the prefix and postfix element multiples. For example
        Input : [1,2,3,4]
        Prefix: [1,2,6,24]
        Postfix: [24,24,12,4]
        Result would be the multiple of prefix and postfix
        [24,12,8,6]. 
        We will try to solve this problem without defining separate array of prefix and postfix.
            '''
            res = [1] * len(nums)
            prefix = 1
            for i in range(len(nums)):
                res[i] = prefix
                prefix *= nums[i]
                
            postfix = 1
            j = len(nums) -1
            while j >= 0:
                res[j] *= postfix
                postfix *= nums[j]
                j-=1
                
            return res
            
```
---
# 35. Find the Duplicate Number
Given an array of integers  `nums`  containing `n + 1`  integers where each integer is in the range  `[1, n]`  inclusive.

There is only  **one repeated number**  in  `nums`, return  _this repeated number_.

You must solve the problem  **without**  modifying the array  `nums` and uses only constant extra space.

**Example 1:**

**Input:** nums = [1,3,4,2,2]
**Output:** 2

**Example 2:**

**Input:** nums = [3,1,3,4,2]
**Output:** 3

**Constraints:**

-   `1 <= n <= 105`
-   `nums.length == n + 1`
-   `1 <= nums[i] <= n`
-   All the integers in  `nums`  appear only  **once**  except for  **precisely one integer**  which appears  **two or more**  times.

**Follow up:**

-   How can we prove that at least one duplicate number must exist in  `nums`?
-   Can you solve the problem in linear runtime complexity?

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Easy way to solve this problem is by using frequency counter dictionary or set but that requires O(n) space
        # However, since it is asked to solve it using O(1) space, we can use Floyd/cycle detection algo to solve this
        
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            
            if slow == fast:
                break
                
        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow
                    
```
---
# 36. Find All Duplicates in an Array
Given an integer array  `nums`  of length  `n`  where all the integers of  `nums`  are in the range  `[1, n]`  and each integer appears  **once**  or  **twice**, return  _an array of all the integers that appears  **twice**_.

You must write an algorithm that runs in `O(n)` time and uses only constant extra space.

**Example 1:**

**Input:** nums = [4,3,2,7,8,2,3,1]
**Output:** [2,3]

**Example 2:**

**Input:** nums = [1,1,2]
**Output:** [1]

**Example 3:**

**Input:** nums = [1]
**Output:** []

**Constraints:**

-   `n == nums.length`
-   `1 <= n <= 105`
-   `1 <= nums[i] <= n`
-   Each element in  `nums`  appears  **once**  or  **twice**.

```python
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        
        for n in nums:
            m = abs(n)
            
            # if negative element appears that means it is seen before
            if nums[m-1] < 0:
                res.append(m)
            else:
                nums[m-1] *= -1 # mark the element as negative
        
        return res

```
---
# 37. Set Matrix Zeroes
Given an  `m x n`  integer matrix  `matrix`, if an element is  `0`, set its entire row and column to  `0`'s.

You must do it  [in place](https://en.wikipedia.org/wiki/In-place_algorithm).

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

**Input:** matrix = [[1,1,1],[1,0,1],[1,1,1]]
**Output:** [[1,0,1],[0,0,0],[1,0,1]]

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)

**Input:** matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
**Output:** [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

**Constraints:**

-   `m == matrix.length`
-   `n == matrix[0].length`
-   `1 <= m, n <= 200`
-   `-231  <= matrix[i][j] <= 231  - 1`

**Follow up:**

-   A straightforward solution using  `O(mn)`  space is probably a bad idea.
-   A simple improvement uses  `O(m + n)`  space, but still not the best solution.
-   Could you devise a constant space solution?

```python
    class Solution:
        def setZeroes(self, matrix: List[List[int]]) -> None:
            """
            Do not return anything, modify matrix in-place instead.
            """
            ROWS, COLS = len(matrix), len(matrix[0])
            rowZero = False
            
            # determine which rows/cols needs to be zeroed
            for r in range(ROWS):
                for c in range(COLS):
                    if matrix[r][c] == 0:
                        matrix[0][c] = 0
                        if r > 0:
                            matrix[r][0] = 0
                        else:
                            rowZero = True
                            
            # print(matrix, rowZero)          
            for r in range(1, ROWS):
                for c in range(1, COLS):
                    if matrix[0][c] == 0 or matrix[r][0] == 0:
                        matrix[r][c] = 0
            
            # If first element is zero make column zero
            if matrix[0][0] == 0:
                for r in range(ROWS):
                    matrix[r][0] = 0
            
            # make row zeros
            if rowZero:
                for c in range(COLS):
                    matrix[0][c] = 0
                
         
```
---
# 38. Spiral Matrix

Given an  `m x n`  `matrix`, return  _all elements of the_  `matrix`  _in spiral order_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

**Input:** matrix = [[1,2,3],[4,5,6],[7,8,9]]
**Output:** [1,2,3,6,9,8,7,4,5]

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)

**Input:** matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
**Output:** [1,2,3,4,8,12,11,10,9,5,6,7]

**Constraints:**

-   `m == matrix.length`
-   `n == matrix[i].length`
-   `1 <= m, n <= 10`
-   `-100 <= matrix[i][j] <= 100`

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)
        result = []
        
        while left < right and top < bottom:
            # get every i in top row
            for i in range(left, right):
                result.append(matrix[left][i])
            top +=1
            
            # get every i in right col
            for i in range(top, bottom):
                result.append(matrix[i][right-1])
            right-=1
            
            # case when there is only one row or one column
            if not( left < right and top < bottom):
                break
            
            # get every i in bottom row
            for i in range(right-1, left-1, -1):
                result.append(matrix[bottom-1][i])
            bottom-=1
            
            # get every i in left col
            for i in range(bottom-1, top-1, -1):
                result.append(matrix[i][left])
            left+=1
            
            
           
            
        return result
            
                                  
```
---
# 39. Rotate Image
You are given an  `n x n`  2D  `matrix`  representing an image, rotate the image by  **90**  degrees (clockwise).

You have to rotate the image  [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly.  **DO NOT**  allocate another 2D matrix and do the rotation.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

**Input:** matrix = [[1,2,3],[4,5,6],[7,8,9]]
**Output:** [[7,4,1],[8,5,2],[9,6,3]]

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)

**Input:** matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
**Output:** [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

**Constraints:**

-   `n == matrix.length == matrix[i].length`
-   `1 <= n <= 20`
-   `-1000 <= matrix[i][j] <= 1000`

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l, r = 0, len(matrix) -1
        
        while l < r:
            for i in range(r-l):
                top, bottom = l, r
                
                # save the top left
                topleft = matrix[top][l + i]
                
                # move bottom left to top left
                matrix[top][l + i] = matrix[bottom - i][l]
                
                # move bottom right to bottom left
                matrix[bottom - i][l] = matrix[bottom][r - i]
                
                # move top right to bottom right
                matrix[bottom][r - i] = matrix[top + i][r]
                
                # move top left to top right
                matrix[top + i][r] = topleft
        
            l+=1
            r-=1
```
---
# 40. Word Search
Given an  `m x n`  grid of characters  `board`  and a string  `word`, return  `true`  _if_  `word`  _exists in the grid_.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

**Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

**Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
**Output:** true

**Example 3:**

![](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)

**Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
**Output:** false

**Constraints:**

-   `m == board.length`
-   `n = board[i].length`
-   `1 <= m, n <= 6`
-   `1 <= word.length <= 15`
-   `board`  and  `word`  consists of only lowercase and uppercase English letters.

**Follow up:**  Could you use search pruning to make your solution faster with a larger  `board`?

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # backtracking
        
        ROWS, COLS = len(board), len(board[0])
        path = set()
        
        def dfs(r, c, i):
            # reached the end of the word
            if i == len(word):
                return True
            
            # out of bound case | letter does not match | repeated path match
            if (r < 0 or c < 0 or r >= ROWS or c >= COLS
               or word[i] != board[r][c]
               or (r, c) in path):
                return False
            
            # found match
            path.add((r,c))
            # check all four adajacent positions
            res = (dfs(r+1, c, i+1) or
                   dfs(r-1, c, i+1) or
                   dfs(r, c + 1, i+1) or
                   dfs(r, c - 1, i+1))
            
            path.remove((r, c)) # cleaning the path
            return res
        
        # O(n * n * 4^n) time complexity
        for r in range(ROWS):
            for c in range(COLS):
                if(dfs(r, c, 0)):
                    return True
        return False
```
---
# 41. Longest Consecutive Sequence
Given an unsorted array of integers  `nums`, return  _the length of the longest consecutive elements sequence._

You must write an algorithm that runs in `O(n)` time.

**Example 1:**

**Input:** nums = [100,4,200,1,3,2]
**Output:** 4
**Explanation:** The longest consecutive elements sequence is `[1, 2, 3, 4]`. Therefore its length is 4.

**Example 2:**

**Input:** nums = [0,3,7,2,5,8,4,6,0,1]
**Output:** 9

**Constraints:**

-   `0 <= nums.length <= 105`
-   `-109  <= nums[i] <= 109`

`LTM DP`

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # if we sort the array then time complexity will be O(nlogn). We need to solve O(n) time
        
        numSet = set(nums)
        longest = 0
        
        for n in nums:
            # Check if it is the start of a sequence. We can determine that by checking if n-1 is there in the set. If it does not exists then it is treated as a start of sequence
            if (n-1) not in numSet:
                l = 0
                while (n + l) in numSet:
                    l+=1
                longest = max(longest, l)
                
        return longest
```
---
# 42. Letter Case Permutation
Given a string  `s`, you can transform every letter individually to be lowercase or uppercase to create another string.

Return  _a list of all possible strings we could create_. Return the output in  **any order**.

**Example 1:**

**Input:** s = "a1b2"
**Output:** ["a1b2","a1B2","A1b2","A1B2"]

**Example 2:**

**Input:** s = "3z4"
**Output:** ["3z4","3Z4"]

**Constraints:**

-   `1 <= s.length <= 12`
-   `s`  consists of lowercase English letters, uppercase English letters, and digits.

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = [""]
        
        for c in s:
            temp = []
            if c.isalpha():
                for o in res:
                    temp.append(o+c.lower())
                    temp.append(o+c.upper())
            else:
                for o in res:
                    temp.append(o+c)
            
            res = temp
            #print(temp, res)
        return res

```
---
# 43.  Subsets
Given an integer array  `nums`  of  **unique**  elements, return  _all possible subsets (the power set)_.

The solution set  **must not**  contain duplicate subsets. Return the solution in  **any order**.

**Example 1:**

**Input:** nums = [1,2,3]
**Output:** [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

**Example 2:**

**Input:** nums = [0]
**Output:** [[],[0]]

**Constraints:**

-   `1 <= nums.length <= 10`
-   `-10 <= nums[i] <= 10`
-   All the numbers of `nums`  are  **unique**.

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # backtracking given [1,2,3]
        '''
            
                        [1]                                    []
                    
                [1, 2]              [1]                 [2]            []
                
        [1,2,3]      [1,2]    [1, 3]    [1]         [2,3]   [2]      [3]    []   
        
        '''
        res = []
        subset = []
            
        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            
            # decision to include nums[i]
            subset.append(nums[i])
            dfs(i+1)
            
            # decision to not include nums[i]
            subset.pop()
            dfs(i+1)
        
        dfs(0)
        return res
            
            
            

```
---
# 44. Subsets II

Given an integer array  `nums`  that may contain duplicates, return  _all possible subsets (the power set)_.

The solution set  **must not**  contain duplicate subsets. Return the solution in  **any order**.

**Example 1:**

**Input:** nums = [1,2,2]
**Output:** [[],[1],[1,2],[1,2,2],[2],[2,2]]

**Example 2:**

**Input:** nums = [0]
**Output:** [[],[0]]

**Constraints:**

-   `1 <= nums.length <= 10`
-   `-10 <= nums[i] <= 10`

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
         # backtracking given [1,2,2,,3]
        '''
            
                        [1]                                         []
                    
                [1, 2]                  [1]                 [2]            []
                
        [1,2,2]         [1,2]      [1, 3]    [1]         [2,3]   [2]      [3]    []   
        
[1,2,2,3]  [1,2,2]  [1,2,3] [1,2]                                         
        '''
        res = []
        nums.sort() #sorting is required so that duplicate elements remain close
        
        def backtrack(i, subset):
            
            if i >= len(nums):
                res.append(subset[::]) #append a copy of subset
                return
            
            # decision to include nums[i]
            subset.append(nums[i])
            backtrack(i+1, subset)
            
            subset.pop()
            # decision to not include nums[i]
            while i+1 < len(nums) and nums[i] == nums[i+1]:
                i+=1  #skip duplicate
            backtrack(i+1, subset)
            
        backtrack(0, [])
        return res
            
                

```
----
# 45. Permutations
Given an array  `nums`  of distinct integers, return  _all the possible permutations_. You can return the answer in  **any order**.

**Example 1:**

**Input:** nums = [1,2,3]
**Output:** [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

**Example 2:**

**Input:** nums = [0,1]
**Output:** [[0,1],[1,0]]

**Example 3:**

**Input:** nums = [1]
**Output:** [[1]]

**Constraints:**

-   `1 <= nums.length <= 6`
-   `-10 <= nums[i] <= 10`
-   All the integers of  `nums`  are  **unique**.

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        
        # base case
        if (len(nums) == 1):
            return [nums[:]] #return list
        
        for i in range(len(nums)):
             # Remove and store first element. Check permutation of remaining elements
            n = nums.pop(0) #[1]
            perms = self.permute(nums)
            
            #perms [2, 3], [3, 2]
            for perm in perms:
                # append popped element [2, 3, 1], [3, 2, 1]
                perm.append(n)
            result.extend(perms) # add it to result
            nums.append(n) # add back popped element
        
        return result
```
---
# 46. Permutations II

Given a collection of numbers,  `nums`, that might contain duplicates, return  _all possible unique permutations  **in any order**._

**Example 1:**

**Input:** nums = [1,1,2]
**Output:**
[[1,1,2],
 [1,2,1],
 [2,1,1]]

**Example 2:**

**Input:** nums = [1,2,3]
**Output:** [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

**Constraints:**

-   `1 <= nums.length <= 8`
-   `-10 <= nums[i] <= 10`

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        perm = []
        count = { n: 0 for n in nums}
        for n in nums:
            count[n] +=1
            
        def dfs():
            if len(perm) == len(nums):
                res.append(perm.copy())
                return
            
            for n in count:
                if count[n] > 0:
                    perm.append(n)
                    count[n] -=1
                
                    dfs()

                    count[n]+=1
                    perm.pop()
                
        dfs()
        return res
```
---
# 47. Combinations

Given two integers  `n`  and  `k`, return  _all possible combinations of_  `k`  _numbers out of the range_  `[1, n]`.

You may return the answer in  **any order**.

**Example 1:**

**Input:** n = 4, k = 2
**Output:**
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

**Example 2:**

**Input:** n = 1, k = 1
**Output:** [[1]]

**Constraints:**

-   `1 <= n <= 20`
-   `1 <= k <= n`

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        
        def backtrack(start, comb):
            if len(comb) == k:
                res.append(comb[::])
                return
            
            for i in range(start, n+1):
                comb.append(i)
                backtrack(i+1, comb)
                comb.pop()
            
        backtrack(1, [])
        return res
```
---
# 48. Combination Sum
Given an array of  **distinct**  integers  `candidates`  and a target integer  `target`, return  _a list of all  **unique combinations**  of_ `candidates` _where the chosen numbers sum to_ `target`_._  You may return the combinations in  **any order**.

The  **same**  number may be chosen from  `candidates`  an  **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is  **guaranteed**  that the number of unique combinations that sum up to  `target`  is less than  `150`  combinations for the given input.

**Example 1:**

**Input:** candidates = [2,3,6,7], target = 7
**Output:** [[2,2,3],[7]]
**Explanation:**
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

**Example 2:**

**Input:** candidates = [2,3,5], target = 8
**Output:** [[2,2,2,2],[2,3,3],[3,5]]

**Example 3:**

**Input:** candidates = [2], target = 1
**Output:** []

**Constraints:**

-   `1 <= candidates.length <= 30`
-   `1 <= candidates[i] <= 200`
-   All elements of  `candidates`  are  **distinct**.
-   `1 <= target <= 500`

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        
        def backtrack(i, curSum, comb):
            # combination sum = target
            if curSum == target:
                res.append(comb[::])
                return
                
            # outbound case
            if curSum > target or i >= len(candidates):
                return
            
            # decision to include candidates[i]
            comb.append(candidates[i])
            backtrack(i, curSum + candidates[i], comb)
            comb.pop()
            # decision to not include candidates[i]
            backtrack(i+1, curSum, comb)
        
        backtrack(0, 0, [])
        return res
```
---
# 49. Combination Sum II

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in  `candidates` where the candidate numbers sum to  `target`.

Each number in  `candidates` may only be used  **once**  in the combination.

**Note:** The solution set must not contain duplicate combinations.

**Example 1:**

**Input:** candidates = [10,1,2,7,6,1,5], target = 8
**Output:** 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

**Example 2:**

**Input:** candidates = [2,5,2,1,2], target = 5
**Output:** 
[
[1,2,2],
[5]
]

**Constraints:**

-   `1 <= candidates.length <= 100`
-   `1 <= candidates[i] <= 50`
-   `1 <= target <= 30`

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort() # sorting to skip duplicates
        
        def backtrack(i, comb, total):
            # base case
            if total == target:
                res.append(comb[::])
                return
            
            # outbound case
            if total > target or i >= len(candidates):
                return
            
            # decision to include candidates[i]
            comb.append(candidates[i])
            backtrack(i+1, comb, total + candidates[i])
            
            comb.pop()
            # skip duplicates
            while (i + 1 < len(candidates)) and (candidates[i] == candidates[i+1]):
                i+=1
           
            # decision to not include candidates[i]
            backtrack(i+1, comb, total)
                
        backtrack(0, [], 0)
        return res
                
            
```
---
# 50. Combination Sum III
Find all valid combinations of  `k`  numbers that sum up to  `n`  such that the following conditions are true:

-   Only numbers  `1`  through  `9`  are used.
-   Each number is used  **at most once**.

Return  _a list of all possible valid combinations_. The list must not contain the same combination twice, and the combinations may be returned in any order.

**Example 1:**

**Input:** k = 3, n = 7
**Output:** [[1,2,4]]
**Explanation:**
1 + 2 + 4 = 7
There are no other valid combinations.

**Example 2:**

**Input:** k = 3, n = 9
**Output:** [[1,2,6],[1,3,5],[2,3,4]]
**Explanation:**
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.

**Example 3:**

**Input:** k = 4, n = 1
**Output:** []
**Explanation:** There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.

**Constraints:**

-   `2 <= k <= 9`
-   `1 <= n <= 60`

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        
        def backtrack(i, curSum, comb):
            
            if len(comb) == k:
                # combination sum = target
                if curSum == n:
                    res.append(comb[::])
                    return
            
            # recursion
            for j in range(i, 9+1):
                # outbound case
                if curSum > n:
                    return
                
                comb.append(j)
                backtrack(j+1, curSum + j, comb)
                comb.pop()
               
        
        backtrack(1, 0, [])
        return res
```
---
# 51. Generate Parentheses
Given  `n`  pairs of parentheses, write a function to  _generate all combinations of well-formed parentheses_.

**Example 1:**

**Input:** n = 3
**Output:** ["((()))","(()())","(())()","()(())","()()()"]

**Example 2:**

**Input:** n = 1
**Output:** ["()"]

**Constraints:**

-   `1 <= n <= 8`

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # if open = N, we cannot add open
        # if close < open, then add close
        
        res = []
        stack = []
        
        def backtrack(open, close):
            if open == close == n:
                res.append("".join(stack))
                return
            
            if open < n:
                stack.append("(")
                backtrack(open + 1, close)
                stack.pop()
                
            if close < open:
                stack.append(")")
                backtrack(open, close+1)
                stack.pop()
                
        backtrack(0,0)
        return res
                
            
```
---
# 52. Target Sum
You are given an integer array  `nums`  and an integer  `target`.

You want to build an  **expression**  out of nums by adding one of the symbols  `'+'`  and  `'-'`  before each integer in nums and then concatenate all the integers.

-   For example, if  `nums = [2, 1]`, you can add a  `'+'`  before  `2`  and a  `'-'`  before  `1`  and concatenate them to build the expression  `"+2-1"`.

Return the number of different  **expressions**  that you can build, which evaluates to  `target`.

**Example 1:**

**Input:** nums = [1,1,1,1,1], target = 3
**Output:** 5
**Explanation:** There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

**Example 2:**

**Input:** nums = [1], target = 1
**Output:** 1

**Constraints:**

-   `1 <= nums.length <= 20`
-   `0 <= nums[i] <= 1000`
-   `0 <= sum(nums[i]) <= 1000`
-   `-1000 <= target <= 1000`

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp = {}  # (index, total) => hash map to store no of ways the target is reached
        
        def backtrack(i, total):
            if i == len(nums):
                return 1 if total == target else 0
            
            if (i , total) in dp:
                return dp[(i, total)]
            
            dp[(i, total)] = (backtrack(i+1, total + nums[i]) +
                             backtrack(i+1, total - nums[i]))
            
            return dp[(i, total)]
        
        return backtrack(0, 0)
```
---
# 53. Palindrome Partitioning
Given a string  `s`, partition  `s`  such that every substring of the partition is a  **palindrome**. Return all possible palindrome partitioning of  `s`.

A  **palindrome**  string is a string that reads the same backward as forward.

**Example 1:**

**Input:** s = "aab"
**Output:** [["a","a","b"],["aa","b"]]

**Example 2:**

**Input:** s = "a"
**Output:** [["a"]]

**Constraints:**

-   `1 <= s.length <= 16`
-   `s`  contains only lowercase English letters.

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        part = []
        
        def dfs(i):
            if i >= len(s):
                res.append(part.copy())
                return
            
            for k in range(i, len(s)):
                if self.isPalindrome(s, i, k):
                    part.append(s[i: k+1])
                    dfs(k+1)
                    part.pop()
                    
        dfs(0)
        return res
    
    def isPalindrome(self, s, l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l, r = l+1, r-1
        
        return True
                    
```
---
# 54. Letter Combinations of a Phone Number

Given a string containing digits from  `2-9`  inclusive, return all possible letter combinations that the number could represent. Return the answer in  **any order**.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![](https://assets.leetcode.com/uploads/2022/03/15/1200px-telephone-keypad2svg.png)

**Example 1:**

**Input:** digits = "23"
**Output:** ["ad","ae","af","bd","be","bf","cd","ce","cf"]

**Example 2:**

**Input:** digits = ""
**Output:** []

**Example 3:**

**Input:** digits = "2"
**Output:** ["a","b","c"]

**Constraints:**

-   `0 <= digits.length <= 4`
-   `digits[i]`  is a digit in the range  `['2', '9']`.

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone_letters = {"2": "abc", 
                         "3": "def", 
                         "4": "ghi", 
                         "5": "jkl", 
                         "6": "mno", 
                         "7": "pqrs", 
                         "8": "tuv", 
                         "9": "wxyz"}
        
        res = []
        
        def backTrack(i, curStr):
            if(len(curStr) == len(digits)):
                res.append(curStr)
                return
            
            for c in phone_letters[digits[i]]:
                backTrack(i+1, curStr+c)
                
        
        if digits:
            backTrack(0, "")
            
        return res         
        
                

```
---
# 55. House Robber
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and  **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array  `nums`  representing the amount of money of each house, return  _the maximum amount of money you can rob tonight  **without alerting the police**_.

**Example 1:**

**Input:** nums = [1,2,3,1]
**Output:** 4
**Explanation:** Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

**Example 2:**

**Input:** nums = [2,7,9,3,1]
**Output:** 12
**Explanation:** Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

**Constraints:**

-   `1 <= nums.length <= 100`
-   `0 <= nums[i] <= 400`


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        #dynamic programming
        rob1, rob2 = 0, 0
        
        # [rob1, rob2, n, n+1, ...]
        for n in nums:
            newRob = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = newRob
        
        return rob2

```
---
# 56. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are  **arranged in a circle.**  That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array  `nums`  representing the amount of money of each house, return  _the maximum amount of money you can rob tonight  **without alerting the police**_.

**Example 1:**

**Input:** nums = [2,3,2]
**Output:** 3
**Explanation:** You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

**Example 2:**

**Input:** nums = [1,2,3,1]
**Output:** 4
**Explanation:** Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

**Example 3:**

**Input:** nums = [1,2,3]
**Output:** 3

**Constraints:**

-   `1 <= nums.length <= 100`
-   `0 <= nums[i] <= 1000`

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # max of only house or skipping first house or skipping last house
        return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))
    
    
    #helper function equivalent to house rob
    def helper(self, nums):
        rob1, rob2 = 0, 0
        
        for n in nums:
            newRob = max(rob1+n, rob2)
            rob1 = rob2
            rob2 = newRob
        
        return rob2

```
---

# 57. Coin Change

You are given an integer array  `coins`  representing coins of different denominations and an integer  `amount`  representing a total amount of money.

Return  _the fewest number of coins that you need to make up that amount_. If that amount of money cannot be made up by any combination of the coins, return  `-1`.

You may assume that you have an infinite number of each kind of coin.

**Example 1:**

**Input:** coins = [1,2,5], amount = 11
**Output:** 3
**Explanation:** 11 = 5 + 5 + 1

**Example 2:**

**Input:** coins = [2], amount = 3
**Output:** -1

**Example 3:**

**Input:** coins = [1], amount = 0
**Output:** 0

**Constraints:**

-   `1 <= coins.length <= 12`
-   `1 <= coins[i] <= 231  - 1`
-   `0 <= amount <= 104`

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dynamic programming bottom up approach using memoization
        dp = [amount+1] * (amount+1) # array size equivalent to amount+1 and value initialized to max value which can be infinite but here we have used amount + 1
        dp[0] = 0
        
        for a in range(1, amount+1):
            for c in coins:
                if a - c >=0:
                    dp[a] = min(dp[a] ,1+ dp[a-c])
        
        return dp[amount] if dp[amount] != amount+1 else -1
```
---

# 58. Maximum Product Subarray

Given an integer array  `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return  _the product_.

The test cases are generated so that the answer will fit in a  **32-bit**  integer.

A  **subarray**  is a contiguous subsequence of the array.

**Example 1:**

**Input:** nums = [2,3,-2,4]
**Output:** 6
**Explanation:** [2,3] has the largest product 6.

**Example 2:**

**Input:** nums = [-2,0,-1]
**Output:** 0
**Explanation:** The result cannot be 2, because [-2,-1] is not a subarray.

**Constraints:**

-   `1 <= nums.length <= 2 * 104`
-   `-10 <= nums[i] <= 10`
-   The product of any prefix or suffix of  `nums`  is  **guaranteed**  to fit in a  **32-bit**  integer.

```python

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = max(nums)
        curMin, curMax = 1, 1
        
        for n in nums:
            tmp = curMax * n
            curMax = max(curMax * n, curMin * n, n)
            curMin = min(tmp, curMin * n, n)
            res = max(curMax, res, curMin)
        
        return res
```
---

# 59. Longest Increasing Subsequence

Given an integer array  `nums`, return the length of the longest strictly increasing subsequence.

A  **subsequence**  is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example,  `[3,6,2,7]`  is a subsequence of the array  `[0,3,1,6,2,2,7]`.

**Example 1:**

**Input:** nums = [10,9,2,5,3,7,101,18]
**Output:** 4
**Explanation:** The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

**Example 2:**

**Input:** nums = [0,1,0,3,2,3]
**Output:** 4

**Example 3:**

**Input:** nums = [7,7,7,7,7,7,7]
**Output:** 1

**Constraints:**

-   `1 <= nums.length <= 2500`
-   `-104  <= nums[i] <= 104`

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        LIS = [1] * len(nums)
        
        for i in range(len(nums)-1, -1, -1):
            for j in range(i+1, len(nums)):
                #print(i, j, nums[i], nums[j], LIS)
                if nums[i] < nums[j]:
                    LIS[i] = max(1 + LIS[j], LIS[i])
                    
        return max(LIS)
```
---
---
#  60. Longest Palindromic Substring
Given a string  `s`, return  _the longest palindromic substring_  in  `s`.

**Example 1:**

**Input:** s = "babad"
**Output:** "bab"
**Explanation:** "aba" is also a valid answer.

**Example 2:**

**Input:** s = "cbbd"
**Output:** "bb"

**Constraints:**

-   `1 <= s.length <= 1000`
-   `s`  consist of only digits and English letters.

`LTM DP`
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        longest = 0
        longestStr = ""
        
        for i in range(len(s)):
                # odd length
                l , r = i, i
                while(l >= 0 and r < len(s) and s[l]==s[r]):
                    if(r-l+1 > longest):
                        longest  = r-l+1
                        longestStr = s[l:r+1]

                    l-=1
                    r+=1
                    
                 # even length
                l , r = i, i+1
                while(l >= 0 and r < len(s) and s[l]==s[r]):
                    if(r-l+1 > longest):
                        longest  = r-l+1
                        longestStr = s[l:r+1]

                    l-=1
                    r+=1
                
        return longestStr
                
            
        
```
---
# 61.  Word Break

Given a string  `s`  and a dictionary of strings  `wordDict`, return  `true`  if  `s`  can be segmented into a space-separated sequence of one or more dictionary words.

**Note**  that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**

**Input:** s = "leetcode", wordDict = ["leet","code"]
**Output:** true
**Explanation:** Return true because "leetcode" can be segmented as "leet code".

**Example 2:**

**Input:** s = "applepenapple", wordDict = ["apple","pen"]
**Output:** true
**Explanation:** Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

**Example 3:**

**Input:** s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
**Output:** false

**Constraints:**

-   `1 <= s.length <= 300`
-   `1 <= wordDict.length <= 1000`
-   `1 <= wordDict[i].length <= 20`
-   `s`  and  `wordDict[i]`  consist of only lowercase English letters.
-   All the strings of  `wordDict`  are  **unique**.

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # dynamic programming
        dp = [False] * (len(s)+1)
        dp[len(s)] = True
        '''
            dp[8] = false
            dp[7] = false
            dp[6] = dp[5] = false
            dp[4] = true ("code" matches)
            dp[3] = dp[2] =dp[1] = false
            dp[0] = dp[0 + len(w)] = dp[4]
        '''
        for i in range(len(s)-1, -1, -1):
            for w in wordDict:
                #print(dp, w)
                if (i+ len(w)) <= len(s) and s[i: i+ len(w)] == w:
                    dp[i] = dp[i+len(w)]
                if dp[i]:
                    break
                
        return dp[0]        
```
---
# 62. Combination Sum IV

```python
```
---

# 63. Decode Ways

```python
```
---

# 64. Unique Paths

```python
```
---

# 65. Jump Game

```python
```
---

# 66. Palindromic Substrings

```python
```
---

# 67. Number of Longest Increasing Subsequence

```python
```
---

# 68. Partition Equal Subset Sum

```python
```
---

# 69. Partition to K Equal Sum Subsets

```python
```
---
# 70. Best Time to Buy and Sell Stock with Cooldown

```python
```
---

# 71. Linked List Cycle II

```python
```
---

# 72. Add Two Numbers

```python
```
---

# 73. Remove Nth Node From End Of List

```python
```
---

# 74. Sort List

```python
```
---
# 75. Reorder List

```python
```
---
# 76. Clone Graph

```python
```
---
# 77. Pacific Atlantic Water Flow

```python
```
---
# 78. Number of Islands

```python
```
---
# 79. Reverse Linked List II

```python
```
---
# 80. Rotate List

```python
```
---
# 81. Swap Nodes in Pairs

```python
```
---
# 82. Odd Even Linked List

```python
```
---
# 83. Kth Smallest Element in a Sorted Matrix

```python
```
---
# 84. Find K Pairs with Smallest Sums

```python
```
---
# 85. Merge Intervals

```python
```
---
# 86. Interval List Intersections

```python
```
---
# 87. Non-overlapping Intervals

```python
```
---
# 88. Task Scheduler

```python
```
---
# 89. Minimum Number of Arrows to Burst Balloons

```python
```
---
# 90. Insert Interval

```python
```
---
# 91. Find Minimum in Rotated Sorted Array

```python
```
---
# 92. Find Peak Element

```python
```
---
# 93. Search in Rotated Sorted Array

```python
```
---
# 94. Search in Rotated Sorted Array II

```python
```
---
# 95. Search a 2D Matrix

```python
```
---
# 96. Search a 2D Matrix II

```python
```
---
# 97. Find K Closest Elements

```python
```
---
# 98. Minimum Size Subarray Sum

```python
```
---
# 99. Fruit Into Baskets

```python
```
---
# 100. Permutation in String

```python
```
---
# 101. Longest Repeating Character Replacement

```python
```
---
# 102. Longest Substring Without Repeating Characters

```python
```
---
# 103. Kth Smallest Element in a BST

```python
```
---
# 104. K Closest Points to Origin

```python
```
---
# 105. Top K Frequent Elements

```python
```
---
# 106. Sort Characters By Frequency

```python
```
---
# 107. Kth Largest Element in an Array

```python
```
---
# 108. Reorganize String 

```python
```
---
# 109. Course Schedule

```python
```
---
# 110. Course Schedule II

```python
```
---
# 111. Minimum Height Trees

```python
```
---
# 112. Sort List

```python
```
---
# 113. Binary Tree Level Order Traversal II

```python
```
---
# 114. Binary Tree Level Order Traversal

```python
```
---
# 115. Binary Tree Zigzag Level Order Traversal

```python
```
---
# 116. Populating Next Right Pointers in Each Node

```python
```
---
# 117. Populating Next Right Pointers in Each Node II

```python
```
---
# 118. Binary Tree Right Side View

```python
```
---
# 119. All Nodes Distance K in Binary Tree

```python
```
---
# 120. Path Sum II

```python
```
---
# 121. Path Sum III

```python
```
---
# 122. Lowest Common Ancestor of a Binary Tree

```python
```
---
# 123. Maximum Binary Tree

```python
```
---
# 124. Maximum Width of Binary Tree


```python
```
---
# 125. Construct Binary Tree from Preorder and Inorder Traversal

```python
```
---
# 126. Validate Binary Search Tree

```python
```
---
# 127. Implement Trie (Prefix Tree)

```python
```
---
# 128. 3 Sum

```python
```
---
# 129. 3 Sum Closest

```python
```
---
# 130. Subarrays with Product Less than K

```python
```
---
# 131. Sort Colours

```python
```
---
# 132. Container With Most Water


```python
```
---
# 133. Longest Word in Dictionary


```python
```
---
# 134. Sort List

```python
```
---
# 135. Maximum XOR of Two Numbers in an Array

```python
```
---
# 136. First Missing Positive

```python
```
---
# 137. Sudoku Solver

```python
```
---
# 138. N-Queens

```python
```
---
# 139. Reverse Nodes in k-Group

```python
```
---
# 140. Merge k Sorted Lists

```python
```
---
# 141. Smallest Range Covering Elements from K Lists

```python
```
---
# 142. Count of Range Sum

```python
```
---
# 143. Sliding Window Maximum

```python
```
---
# 144. Minimum Number of K Consecutive Bit Flips

```python
```
---
# 145. Count Unique Characters of All Substrings of a Given String

```python
```
---
# 146. Minimum Window Substring

```python
```
---
# 147. Substring with Concatenation of All Words

```python
```
---
# 148. Course Schedule III

```python
```
---
# 149. Maximum Frequency Stack


```python
```
---
# 150. Binary Tree Maximum Path Sum

```python
```
---
# 151. Serialize and Deserialize Binary Tree

```python
```
---
# 152. Word Search II

```python
```
---
# 153. Find Median from Data Stream

```python
```
---
# 154. Sliding Window Median

```python
```
---
# 155. Trapping Rain Water


```python
```
---
# 156. Concatenated Words

```python
```
---
# 157. Prefix and Suffix Search

```python
```
---
# 158. Palindrome Pairs

```python
```
---
# 159. Sort Items by Groups Respecting Dependencies

```python
```
---
# 160. Median of Two Sorted Arrays

```python
```
---
