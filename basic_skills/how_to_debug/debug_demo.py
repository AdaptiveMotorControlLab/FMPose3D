def calculate_average(numbers):
    total_sum = 0
    for num in numbers:
        total_sum += num
    average = total_sum / len(numbers) 
    return average

def main():
    # List of numbers to calculate the average
    numbers = [1,2,3,4]
    
    # Call the calculate_average function
    avg = calculate_average(numbers)
    
    # Print the result
    print(f"The average of {numbers} is {avg}")

if __name__ == "__main__":
    main()