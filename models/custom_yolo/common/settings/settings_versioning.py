import hashlib
from datetime import datetime


class SettingsVersionControl:
    def __init__(self):
        self.history = []

    def commit(self, changes):
        version_info = {
            'timestamp': datetime.now(),
            'changes': changes,
            'hash': self._generate_hash(changes)
        }

    def rollback(self, steps: int = 1):
        # rollback to previous version
        pass

    def _generate_hash(self, changes):
        return hashlib.md5(str(changes).encode('utf-8')).hexdigest()
