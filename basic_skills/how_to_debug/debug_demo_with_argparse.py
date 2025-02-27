import argparse
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Handle the empty list case
    total_sum = 0
    for num in numbers:
        total_sum += num
    average = total_sum / len(numbers)
    return average

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Calculate the average of a list of numbers.")    
    # Add a positional argument to accept a list of numbers
    parser.add_argument('--numbers', metavar='N', type=float, 
        nargs='+',  # Accept one or more numbers
        default=[1, 2, 3, 4],  # Default list of numbers
        help="A list of numbers to calculate the average"
    )
    # Parse the arguments
    args = parser.parse_args()
    # Calculate average
    avg = calculate_average(args.numbers)
    
    # Print the result
    print(f"The average of {args.numbers} is {avg}")

if __name__ == "__main__":
    main()