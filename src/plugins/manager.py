from .base import Plugin

class PluginManager:
    def __init__(self):
        self._plugins: dict[str, Plugin] = {}

    def register(self, name: str, plugin_cls: type[Plugin]):
        self._plugins[name] = plugin_cls()

    def discover_plugins(self):
        for plugin_cls in Plugin.discover():
            self.register(plugin_cls.__name__, plugin_cls)

    def run(self, name: str, context: dict):
        plugin = self._plugins.get(name)
        if not plugin:
            raise KeyError(f"Plugin '{name}' not registered")
        return plugin.execute(context)
