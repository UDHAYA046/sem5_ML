#wap to generate n random numbers and calculate - mean , median , mode
import random  # To generate random numbers

# Function to generate a list of random integers
def generate_random_list(count):
    numbers = []  # Empty list to hold generated numbers

    for i in range(count):
        # Generate a random number between 1 and 100 (you can adjust range)
        num = random.randint(1, 100)
        numbers.append(num)  # Add to list

    return numbers  # Return the full list


# Function to calculate the mean (average) of a list
def calculate_mean(data):
    total = 0

    for value in data:
        total = total + value  # Add each number

    mean_value = total / len(data)  # Total divided by number of items
    return mean_value


# Function to calculate the median of a list
def calculate_median(data):
    sorted_data = sorted(data)  # Sort the data in ascending order
    n = len(sorted_data)

    # Check if the number of elements is odd or even
    if n % 2 == 1:
        median_value = sorted_data[n // 2]  # Middle element for odd
    else:
        middle1 = sorted_data[n // 2 - 1]
        middle2 = sorted_data[n // 2]
        median_value = (middle1 + middle2) / 2  # Average of two middle elements

    return median_value


# Function to calculate the mode of a list
def calculate_mode(data):
    max_count = 0
    mode_value = data[0]

    for i in data:
        count = 0
        for j in data:
            if i == j:
                count = count + 1

        # Update mode if this number appears more times
        if count > max_count:
            max_count = count
            mode_value = i

    return mode_value


# --- Main Program ---

# Ask user how many random numbers to generate
print("Enter how many random numbers to generate:")
n = int(input())

# Generate the random list
numbers = generate_random_list(n)

# Display the list
print("Generated random numbers:")
for num in numbers:
    print(num, end=" ")
print()  # For new line

# Calculate mean, median, and mode
mean = calculate_mean(numbers)
median = calculate_median(numbers)
mode = calculate_mode(numbers)

# Display results
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
5
