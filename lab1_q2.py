# Function to read a matrix using nested loops
def get_matrix(rows, cols, name):
    matrix = []  # This list will store the entire matrix

    print("Enter elements for Matrix", name, ":")  # Prompt user for matrix input

    for i in range(rows):  # Loop over the number of rows
        row = []  # Temporary list to store the current row

        print("Row", i + 1, ":")  # Show which row is being entered (1-based for user)
        for j in range(cols):  # Loop over the number of columns
            print("Enter element at position [", i + 1, "][", j + 1, "]:", end=" ")
            val = int(input())  # Get integer input from user
            row.append(val)  # Add the value to the current row

        matrix.append(row)  # Add the completed row to the matrix

    return matrix  # Return the final 2D matrix


# Function to multiply two matrices
def multiply_matrices(a, b):
    result = []  # This will store the resulting matrix after multiplication

    for i in range(len(a)):  # Loop through rows of matrix A
        row = []  # Temporary list to store the resulting row

        for j in range(len(b[0])):  # Loop through columns of matrix B
            sum = 0  # Initialize the element at position [i][j] of the result

            for k in range(len(a[0])):  # Loop through columns of A / rows of B
                # Multiply corresponding elements and add to sum
                sum = sum + a[i][k] * b[k][j]

            row.append(sum)  # Add the calculated value to the current row

        result.append(row)  # Add the completed row to the result matrix

    return result  # Return the product matrix


# --- Main Program Starts Here ---

# Accept matrix dimensions from the user
p = int(input("no of rows in matrix A : "))
q = int(input("no of columns in matrix A : "))
r = int(input("no of rows in matrix B : "))
s = int(input("no of columns in matrix B : "))

# Check if matrix multiplication is possible
if q != r:
    print("Error! , multiplication is not possible . ")
else:
    # Read both matrices from the user
    MA = get_matrix(p, q, "A")
    MB = get_matrix(r, s, "B")

    # Multiply the matrices
    product = multiply_matrices(MA, MB)

    # Display the result
    print("Product of Matrix A and Matrix B:")
    for i in range(len(product)):  # Loop over each row
        for j in range(len(product[0])):  # Loop over each column
            print(product[i][j], end=" ")  # Print each element with a space
        print()  # Newline after each row
