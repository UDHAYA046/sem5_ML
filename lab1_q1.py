# wap to count the number of vowels and consonants

# Function to evaluate how many vowels and consonants are in the given text
def chk(word):
    vowels = "aeiou"  # String of all vowels
    alphabets = "abcdefghijklmnopqrstuvwxyz"  # All lowercase letters

    vowel_count = 0       # To count number of vowels
    consonant_count = 0   # To count number of consonants

    for i in word.lower():  # Convert input to lowercase and loop through each character
        if i in vowels:     # Check if character is a vowel
            vowel_count += 1
        elif i in alphabets and i not in vowels:
            # If character is an alphabet but not a vowel → it's a consonant
            consonant_count += 1

    # Return both counts as a tuple
    return vowel_count, consonant_count

# --- Main block ---

# Take input from the user
wrd = input("Enter a Word: ")

# Call the function and store the result
v, c = chk(wrd)

# Print the results
print("Vowel count:", v)
print("Consonant count:", c)
