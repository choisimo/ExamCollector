from PyQt5.QtCore import QObject, pyqtSignal

from models.custom_yolo.core.services.db_service import DBService


class SettingsManager(QObject):
    setting_changed = pyqtSignal(str, object)
    """
    DB -> cache -> code
    특정 설정이 바뀌면 setting_changed(key_path, new_value) 시그널 발생
    """
    def __init__(self):
        super().__init__()
        self._cache = {}
        self.db = DBService()
        self._load_initial_settings()

    def _load_initial_settings(self):
        """
        DB에서 설정을 불러와 캐시에 저장합니다.
        :return: None
        """
        training_config = self.db.get_training_config()
        labeling_config = self.db.get_labeling_config()
        document_config = self.db.get_document_converter()

        self._cache['training'] = training_config
        self._cache['labeling'] = labeling_config
        self._cache['document'] = document_config

    def get(self, section, key, default=None):
        # ex) get('training', 'device')
        return self._cache.get(section, {}).get(key, default)

    def update(self, section, key, value):
        self._cache.setdefault(section, {})[key] = value
        self.setting_changed.emit(f"{section}.{key}", value)

    def save_all(self):
        """
        캐시에 저장된 설정을 DB에 저장합니다.
        :return:
        """
        train_config = self._cache.get('training', None)
        if train_config and 'id' in train_config:
            self.db.update_training_config(train_config['id'], train_config)

        label_config = self._cache.get('labeling', None)
        if label_config and 'id' in label_config:
            self.db.update_labeling_config(label_config['id'], label_config)

        doc_config = self._cache.get('document', None)
        if doc_config and 'id' in doc_config:
            self.db.update_document_converter(doc_config['id'], doc_config)

