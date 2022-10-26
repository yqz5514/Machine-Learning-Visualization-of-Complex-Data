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
nums = [2,0,2,1,1,0]
class Solution:
    def sortColors(self, nums):
        red, white, blue = 0, 0, len(nums)-1
    
        while white <= blue: # for numbers not in list, position will return 1
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red] # swap index 
                white += 1
                red += 1
                #print(nums[red], nums[white],nums[blue],white,red, blue)
            elif nums[white] > nums[blue]:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
                #print(nums[red], nums[white],nums[blue],white,red, blue)
            else:
                white += 1
                #print(nums[red], nums[white],nums[blue],white,red, blue)
#%%
def sortColors(nums):
        red, white, blue = 0, 0, len(nums)-1
        print(nums[red], nums[white],nums[blue],white,red, blue)
    
        while white <= blue: # for numbers not in list, position will return 1
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red] # swap index 
                white += 1
                red += 1
                print('1st',nums[red], nums[white],nums[blue],white,red, blue,nums)
            elif nums[white] > nums[blue]:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
                print('2nd',nums[red], nums[white],nums[blue],white,red, blue,nums)
            else:
                white += 1
                print('3rd',nums[red], nums[white],nums[blue],white,red, blue,nums)

#%%
nums = [2,0,2,1,1,0]

sortColors(nums)

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
 #return sorted(set(nums), reverse = True)[2] if len(set(nums)) >= 3 else max(nums)
 # 
#%%
# #26. Remove Duplicates from Sorted Array
#Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

#Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

#Return k after placing the final result in the first k slots of nums.

#Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

#Custom Judge:

#The judge will test your solution with the following code:

#int[] nums = [...]; // Input array
#int[] expectedNums = [...]; // The expected answer with correct length

#int k = removeDuplicates(nums); // Calls your implementation

#assert k == expectedNums.length;
#for (int i = 0; i < k; i++) {
#    assert nums[i] == expectedNums[i];
#}
#If all assertions pass, then your solution will be accepted.

 

#Example 1:

#Input: nums = [1,1,2]
#Output: 2, nums = [1,2,_]
#Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
#It does not matter what you leave beyond the returned k (hence they are underscores).
#Example 2:

#Input: nums = [0,0,1,1,1,2,2,3,3,4]
#Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
#Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
#It does not matter what you leave beyond the returned k (hence they are underscores).
 

#Constraints:

#1 <= nums.length <= 3 * 104
#-100 <= nums[i] <= 100
#nums is sorted in non-decreasing order. 

#%%
class Solution(object):
    def removeDuplicates(self, nums):
        nums[:] = sorted(set(nums)) # num[:] same as num.copy()
        return len(nums)
    
    

#%%
# #27. Remove Element
#Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

#Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

#Return k after placing the final result in the first k slots of nums.

#Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

#Custom Judge:

#The judge will test your solution with the following code:

#int[] nums = [...]; // Input array
#int val = ...; // Value to remove
#int[] expectedNums = [...]; // The expected answer with correct length.
#                            // It is sorted with no values equaling val.

#int k = removeElement(nums, val); // Calls your implementation

#assert k == expectedNums.length;
#sort(nums, 0, k); // Sort the first k elements of nums
#for (int i = 0; i < actualLength; i++) {
#    assert nums[i] == expectedNums[i];
#}
#If all assertions pass, then your solution will be accepted.

 

#Example 1:

#Input: nums = [3,2,2,3], val = 3
#Output: 2, nums = [2,2,_,_]
#Explanation: Your function should return k = 2, with the first two elements of nums being 2.
#It does not matter what you leave beyond the returned k (hence they are underscores).

#%%
def removeElement(self, nums, val):
    i = 0
    for x in nums:
        if x != val:
            nums[i] = x
            i += 1
    return i

#%%
# fast pointer, slow pointer
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if nums is None or len(nums)==0: 
            return 0 
        l=0
        r=len(nums)-1
        while l<r: 
            while(l<r and nums[l]!=val):
                l+=1
            while(l<r and nums[r]==val):
                r-=1
            nums[l], nums[r]=nums[r], nums[l]
        print(nums)
        if nums[l]==val: 
            return l 
        else: 
            return l+1
        
#%%
# #283 move zeros
#Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

#Note that you must do this in-place without making a copy of the array.

#Example 1:

#Input: nums = [0,1,0,3,12]
#Output: [1,3,12,0,0]
#Example 2:

#Input: nums = [0]
#Output: [0]
 

#Constraints:

#1 <= nums.length <= 104
#-231 <= nums[i] <= 231 - 1

class Solution:
    def moveZeroes(self, nums: list) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]

            # wait while we find a non-zero element to
            # swap with you
            if nums[slow] != 0:
                slow += 1
        