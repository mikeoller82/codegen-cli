def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a (float): First number
        b (float): Second number
    
    Returns:
        float: Sum of a and b
    """
    return a + b

def main():
    # Test the function
    result1 = add_numbers(5, 3)
    result2 = add_numbers(10.5, 2.5)
    result3 = add_numbers(-1, 1)
    
    print(f"5 + 3 = {result1}")
    print(f"10.5 + 2.5 = {result2}")
    print(f"-1 + 1 = {result3}")

if __name__ == "__main__":
    main()