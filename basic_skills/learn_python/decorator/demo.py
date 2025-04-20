def f1(func):
    def wrapper(*args, **kwargs):
        print("Started")
        return func(*args, **kwargs)
        print("Ended")
    
    return wrapper

@f1
def f(a, b=9):
    print(a, b)

def add(x, y):
    return x + y

f(1,3)