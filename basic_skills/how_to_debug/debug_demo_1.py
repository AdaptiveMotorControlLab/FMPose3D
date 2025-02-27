def calculate_average(numbers):
    total_sum = 0
    for num in numbers:
        total_sum += num
    average = total_sum / len(numbers)  # Bug: What happens if the list is empty?
    return average

def main():
    # List of numbers to calculate the average
    numbers = [10, 20, 30, 40, 50]  # Try changing this to an empty list []
    
    # Call the calculate_average function
    avg = calculate_average(numbers)
    
    # Print the result
    print(f"The average of {numbers} is {avg}")

if __name__ == "__main__":
    main()