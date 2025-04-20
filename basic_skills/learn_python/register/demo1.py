from register import Register

my_register = Register()


@my_register
def add(a, b):
    return a + b


@my_register
def multiply(a, b):
    return a * b


@my_register('matrix multiply')
def mul_multiply(a, b):
    pass

if __name__ == '__main__':
    # check register information
    for k, v in my_register.items():
        print(f"key: {k}, value: {v}")
    print()
    