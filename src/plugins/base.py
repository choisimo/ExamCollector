import pkgutil
import importlib
import os
from abc import ABC, abstractmethod

class Plugin(ABC):
    @abstractmethod
    def execute(self, context: dict):
        pass

    @classmethod
    def discover(cls) -> list[type]:
        discovered = []
        package_dir = os.path.dirname(__file__)
        package_name = __package__
        for finder, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            full_name = f"{package_name}.{module_name}"
            mod = importlib.import_module(full_name)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, cls) and obj is not cls:
                    discovered.append(obj)
        return discovered
