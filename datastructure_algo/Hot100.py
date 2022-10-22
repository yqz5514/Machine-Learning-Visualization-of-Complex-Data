#%%[Markdown]
# ## Basic Array problems(3)
##########################################################
#%%
# ##485 Max Consecutive Ones
#Given a binary array nums, return the maximum number of consecutive 1's in the array.
#Example 1:
#Input: nums = [1,1,0,1,1,1]
#Output: 3
#Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
#Example 2:
#Input: nums = [1,0,1,1,0,1]
#Output: 2
#Constraints:
#1 <= nums.length <= 105
#nums[i] is either 0 or 1.
#%%
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        cnt = 0
        ans = 0
        for num in nums:
            if num == 1:
                cnt += 1
                ans = cnt if ans < cnt else ans 
            else:
                cnt = 0
        return ans
    
#%%
####75#### sort colors/ dutch national flag problem[M]
#Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

#We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

#You must solve this problem without using the library's sort function.

#Example 1:

#Input: nums = [2,0,2,1,1,0]
#Output: [0,0,1,1,2,2]
#Example 2:

#Input: nums = [2,0,1]
#Output: [0,1,2]

#Constraints:
#n == nums.length
#1 <= n <= 300
#nums[i] is either 0, 1, or 2.
#%%
class Solution:
    def sortColors(self, nums):
        red, white, blue = 0, 0, len(nums)-1
    
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red] # swap them 
                white += 1
                red += 1
            elif nums[white] > nums[blue]:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
            else:
                white += 1
#%%
x = [2,3,4,1,1,1,1]
x[5]
#%%[Markdown]
# 