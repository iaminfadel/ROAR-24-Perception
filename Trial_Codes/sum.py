def find_combinations(nums, target):
    def backtrack(start, target, path):
        nonlocal shortest_combination
        if target == 0:
            result.append(path)
            if len(path) < len(shortest_combination):
                shortest_combination = path[:]
            return
        if target < 0 or start == len(nums):
            return
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:  # Skip duplicates
                continue
            backtrack(i + 1, target - nums[i], path + [nums[i]])

    nums.sort()  # Sort the numbers to handle duplicates
    result = []
    shortest_combination = []
    backtrack(0, target, [])
    if result:
        shortest_combination = min(result, key=len)
        return shortest_combination , result
    else:
        return "No combination found."

# Example usage:
label_with_coordinates = {'5': (406, 46), '3': (406, 407), '8': (165, 407), '1': (167, 45)}

# Extract keys into a list
labels_list = list(label_with_coordinates.keys())

# Convert each label to an integer
labels_as_numbers = [int(label) for label in labels_list]

print(labels_as_numbers)

# numbers = [1, 3, 5, 8]
target = 8
shortest_combination , result = find_combinations(labels_as_numbers, target)
print("Shortest Combination:", shortest_combination)
print("Result Combination:", result)
