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
	
        class Solution:
	        def containsDuplicate(self, nums: List[int]) -> bool:
                lookup = {}
                
                for i in range(len(nums)):
                    if nums[i] in lookup:
                        return True
                    else:
                        lookup[nums[i]] = 1
                        
                return False

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

    class Solution:
        def missingNumber(self, nums: List[int]) -> int:
            lookup = set()
            
            for n in nums:
                lookup.add(n)
                
            for i in range(len(nums)+1):
                if not i in lookup:
                    return i
                
            return None

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

    class Solution:
        def climbStairs(self, n: int) -> int:
        
            one, two = 1, 1
            
            # dynamic programming bottom up approach
            for i in range(1, n):
                temp = one
                one = one + two
                two = temp
                
            return one

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

---
        
