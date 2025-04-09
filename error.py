# Define a variable with a string type
x = "5"

# Try to add a string to an integer, which will cause a TypeError
try:
    result = x + 5  # This will cause a TypeError since x is a string and 5 is an integer
except TypeError as e:
    print(f"Error occurred: {e}")

