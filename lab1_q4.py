#wap to take matrix as input and return its transpose

# Function to read a matrix from user input using nested loops
def get_matrix(rows, cols):
    matrix = []  # This will store the entire matrix as a list of lists

    print("Enter the elements of the matrix:")

    for i in range(rows):  # Loop through each row
        row = []  # Temporary list to store elements of the current row
        print("Row", i + 1, ":")

        for j in range(cols):  # Loop through each column
            # Ask the user to enter an element for position [i][j]
            print("Enter element at position [", i + 1, "][", j + 1, "]:", end=" ")
            value = int(input())  # Convert the input to an integer
            row.append(value)  # Add the element to the current row

        matrix.append(row)  # Add the completed row to the matrix

    return matrix  # Return the completed matrix


# Function to compute and return the transpose of the matrix
def transpose_matrix(matrix):
    # The transpose of a matrix flips rows and columns:
    # Row i becomes Column i and Column j becomes Row j

    transposed = []  # This will store the transposed matrix

    # Outer loop iterates over columns of the original matrix
    for j in range(len(matrix[0])):  # len(matrix[0]) gives number of columns
        row = []  # This will form a row in the transposed matrix

        # Inner loop iterates over rows of the original matrix
        for i in range(len(matrix)):  # len(matrix) gives number of rows
            row.append(matrix[i][j])  # Add the transposed element

        transposed.append(row)  # Add the new row to the transposed matrix

    return transposed  # Return the transposed matrix


# --- Main Program Starts Here ---

# Ask the user to enter the number of rows
print("Enter number of rows:")
rows = int(input())

# Ask the user to enter the number of columns
print("Enter number of columns:")
cols = int(input())

# Call the function to read the matrix from the user
A = get_matrix(rows, cols)

# Call the function to compute the transpose
T = transpose_matrix(A)

# Display the result
print("Transpose of the matrix:")

# Loop through each row in the transposed matrix
for i in range(len(T)):
    # Loop through each column in the row
    for j in range(len(T[0])):
        print(T[i][j], end=" ")  # Print each element with a space
    print()  # Move to the next line after each row
