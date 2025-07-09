#wap to find the number of common elements in a list of integers

# Function to find common elements without duplicates
def common_elements(list1, list2):
    result = []
    for i in list1:
        for j in list2:
            if i == j and i not in result:
                result.append(i)
    return result

# --- Main Program ---

print("Enter first list of integers (comma separated):")
raw1 = input()  # like 1,2,3,4,5

print("Enter second list of integers (comma separated):")
raw2 = input()  # like 4,5,6

# Convert to integer lists
list1 = []
part1 = raw1.split(",")
for val in part1:
    list1.append(int(val))

list2 = []
part2 = raw2.split(",")
for val in part2:
    list2.append(int(val))

# Find common elements
common = common_elements(list1, list2)

# Display common elements
print("Common elements are:")
for value in common:
    print(value, end=" ")


    
