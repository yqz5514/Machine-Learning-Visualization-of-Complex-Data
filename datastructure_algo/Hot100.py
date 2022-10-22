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
    
        while white <= blue: # for numbers not in list, position will return 1
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red] # swap index 
                white += 1
                red += 1
            elif nums[white] > nums[blue]:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
            else:
                white += 1
#%%
x = [2,3,4,1,1,1,1]
x[1]
#%%
###414###Third Maximum Number
#Given an integer array nums, return the third distinct maximum number in this array. If the third maximum does not exist, return the maximum number.

#Example 1:

#Input: nums = [3,2,1]
#Output: 1
#Explanation:
#The first distinct maximum is 3.
#The second distinct maximum is 2.
#The third distinct maximum is 1.
#Example 2:

#Input: nums = [1,2]
#Output: 2
#Explanation:
#The first distinct maximum is 2.
#The second distinct maximum is 1.
#The third distinct maximum does not exist, so the maximum (2) is returned instead.
#Example 3:

#Input: nums = [2,2,3,1]
#Output: 1
#Explanation:
#The first distinct maximum is 3.
#The second distinct maximum is 2 (both 2's are counted together since they have the same value).
#The third distinct maximum is 1.

#Constraints:

#1 <= nums.length <= 104
#-231 <= nums[i] <= 231 - 1

#%%

def thirdMax(self, nums):
        nums = set(nums)
        if len(nums) < 3:
            return max(nums)
        nums.remove(max(nums))
        nums.remove(max(nums))
        return max(nums)
    
#%%
 return sorted(set(nums), reverse = True)[2] if len(set(nums)) >= 3 else max(nums)
#%%[Markdown]
# 