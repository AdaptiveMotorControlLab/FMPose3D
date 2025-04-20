# -*- coding: utf-8 -*-
# @file: register.py
class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target):
        """
        The __call__ method allows an instance of the class to be called as if it were a function. for example:
        obj  = MyClass()
        obj(target) # This will internally call obj.__call__(target)
        """
        return self.register(target)

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value

        if callable(target):    # 传入的target可调用 --> 没有给注册名 --> 传入的函数名或类名作为注册名
            return add_item(target.__name__, target)
        else:                   # 不可调用 --> 传入了注册名 --> 作为可调用对象的注册名
            return lambda x: add_item(target, x)

    def __setitem__(self, key, value):
        """
        This method is called when you assign a value to a key in the object, like you would with a dictionary.
        example:
            ```python 
            obj = MyClass()
            obj[key] = value  # Internally calls obj.__setitem__(key, value)
            ```
        """
        self._dict[key] = value

    def __getitem__(self, key):
        """
        This method is called when you access a value using a key, like you would with a dictionary.
        value = obj[key]  # Internally calls obj.__getitem__(key)
        """
        return self._dict[key]

    def __contains__(self, key):
        """
        This method is called when you use the in keyword to check if a key exists in the object.
        example:
            ```python
            if key in obj:
                # Do something 
            ```
        """
        return key in self._dict

    def __str__(self):
        """
        This method is called when you convert the object to a string using str() or print().
        example:
            print(obj)  # Internally calls obj.__str__()
        """
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()