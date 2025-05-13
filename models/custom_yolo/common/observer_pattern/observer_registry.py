from collections import defaultdict


"""
Observer-pattern register and notify class
=> register: register a callback function to a key_path
=> notify: notify all callback functions registered to a key_path
"""


class ObserverRegistry:
    _observers = defaultdict(list)

    @classmethod
    def register(cls, key_path, callback):
        cls._observers[key_path].append(callback)

    @classmethod
    def notify(cls, key_path, value):
        for cb in cls._observers.get(key_path, []):
            cb(value)
