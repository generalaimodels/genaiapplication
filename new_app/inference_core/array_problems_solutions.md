# Array Problems Solutions

**Generated**: 2025-12-27 16:59:12
**Framework**: SOTA Agentic Framework 3.0
**Problems Solved**: 5/5

---

## Problem 1: Two Sum

**Description**: Find two numbers in array that add up to target

**Example**: Input: [2, 7, 11, 15], target=9. Output: [0, 1]

**Constraints**: Array length ≤ 10^4, unique solution exists

### Solution

```python
# Two Sum - O(n) with hash map
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Test
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(f"Input: {nums}, Target: {target}")
print(f"Output: {result}")
print(f"Verification: {nums[result[0]]} + {nums[result[1]]} = {nums[result[0]] + nums[result[1]]}")
```

### Output

```
Output:
Input: [2, 7, 11, 15], Target: 9
Output: [0, 1]
Verification: 2 + 7 = 9

```

**Status**: ✅ Success

---

## Problem 2: Maximum Subarray Sum

**Description**: Find contiguous subarray with largest sum (Kadane's algorithm)

**Example**: Input: [-2,1,-3,4,-1,2,1,-5,4]. Output: 6 (subarray [4,-1,2,1])

**Constraints**: Array length ≤ 10^5

### Solution

```python
# Kadane's Algorithm - O(n)
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Test
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = max_subarray(nums)
print(f"Input: {nums}")
print(f"Maximum Subarray Sum: {result}")
```

### Output

```
Output:
Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Maximum Subarray Sum: 6

```

**Status**: ✅ Success

---

## Problem 3: Rotate Array

**Description**: Rotate array to the right by k steps

**Example**: Input: [1,2,3,4,5,6,7], k=3. Output: [5,6,7,1,2,3,4]

**Constraints**: In-place with O(1) extra space

### Solution

```python
# Rotate Array - O(n) time, O(1) space
def rotate(nums, k):
    n = len(nums)
    k = k % n
    
    def reverse(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    reverse(nums, 0, n - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, n - 1)
    return nums

# Test
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
print(f"Original: {nums}")
result = rotate(nums.copy(), k)
print(f"Rotated by {k}: {result}")
```

### Output

```
Output:
Original: [1, 2, 3, 4, 5, 6, 7]
Rotated by 3: [5, 6, 7, 1, 2, 3, 4]

```

**Status**: ✅ Success

---

## Problem 4: Remove Duplicates from Sorted Array

**Description**: Remove duplicates in-place, return new length

**Example**: Input: [1,1,2,2,3]. Output: 3, array=[1,2,3,_,_]

**Constraints**: Must be in-place, O(1) extra space

### Solution

```python
# Remove Duplicates - Two Pointers O(n)
def remove_duplicates(nums):
    if not nums:
        return 0
    
    write_idx = 1
    for read_idx in range(1, len(nums)):
        if nums[read_idx] != nums[read_idx - 1]:
            nums[write_idx] = nums[read_idx]
            write_idx += 1
    return write_idx

# Test
nums = [1, 1, 2, 2, 3, 3, 3, 4]
print(f"Original: {nums}")
length = remove_duplicates(nums)
print(f"New length: {length}")
print(f"Modified array: {nums[:length]}")
```

### Output

```
Output:
Original: [1, 1, 2, 2, 3, 3, 3, 4]
New length: 4
Modified array: [1, 2, 3, 4]

```

**Status**: ✅ Success

---

## Problem 5: Merge Sorted Arrays

**Description**: Merge two sorted arrays into first array

**Example**: Input: nums1=[1,2,3,0,0,0], m=3, nums2=[2,5,6], n=3. Output: [1,2,2,3,5,6]

**Constraints**: nums1 has size m+n, merge in-place

### Solution

```python
# Merge Sorted Arrays - O(m+n)
def merge(nums1, m, nums2, n):
    p1, p2 = m - 1, n - 1
    write_idx = m + n - 1
    
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[write_idx] = nums1[p1]
            p1 -= 1
        else:
            nums1[write_idx] = nums2[p2]
            p2 -= 1
        write_idx -= 1
    return nums1

# Test
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3
print(f"nums1: {nums1[:m]}, nums2: {nums2}")
result = merge(nums1, m, nums2, n)
print(f"Merged: {result}")
```

### Output

```
Output:
nums1: [1, 2, 3], nums2: [2, 5, 6]
Merged: [1, 2, 2, 3, 5, 6]

```

**Status**: ✅ Success

---

## Complexity Analysis

| Problem | Time Complexity | Space Complexity | Algorithm |
|---------|----------------|------------------|-----------|
| Two Sum | O(n) | O(n) | Hash Map |
| Max Subarray | O(n) | O(1) | Kadane's Algorithm |
| Rotate Array | O(n) | O(1) | Triple Reverse |
| Remove Duplicates | O(n) | O(1) | Two Pointers |
| Merge Sorted Arrays | O(m+n) | O(1) | Two Pointers |

---

## Framework Features Demonstrated

- ✅ **ReAct Agent**: Reasoning + Acting pattern
- ✅ **Tool Integration**: Python code execution
- ✅ **Result Handling**: Robust error handling with Result types
- ✅ **Automated Documentation**: Generated markdown reports
- ✅ **Best Practices**: Optimal algorithms with complexity guarantees

## Running This Demo

```bash
cd /home/test1/Hemanth/Agentic_Framework/inference_core
python -m Agentic_core.examples.array_problem_solver
```

**Generated by**: SOTA Agentic Framework 3.0