# -*- coding: utf-8 -*-
# @file: class_register.py

class Register(dict):
    def __init__(self, registry_name, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            print("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def __call__(self, target):
        return self.register(target)

    def register(self, target):
        """Decorator to register a function or class."""
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def items(self):
        return self._dict.items()