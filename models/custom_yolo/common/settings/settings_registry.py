from PyQt5.QtWidgets import QButtonGroup

class SettingsCore:

    def __init__(self, db_service):
        self.db = db_service

    def load_settings(self):
        data = {}
        training_settings = self.db.get_training_config()
        labeling_settings = self.db.get_labeling_config()
        data['training'] = training_settings
        data['labeling'] = labeling_settings
        return data

    def save_training_config(self, new_config):
        row = self.db.get_training_config()
        if row:
            self.db.update_training_config(row['id'], new_config['data_yaml'])
        else:
            self.db.insert_training_config(
                epochs=50, batch_size=16, device="cpu", imgsz=640,
                YOLO_model="yolov8n.pt", custom_model_path="", data_yaml=path,
                data_config_id=1, results_config_id=1, model_paths_id=1
            )


class SettingsUI:
    """
    Settings UI를 구성하는 클래스입니다.
    """
    def __init__(self):
        self.rb_cpu = None
        self.rb_cuda = None
        self.rb_hip = None

        self.device_group = QButtonGroup()

    def setup_device_radios(self):
        for btn in [self.rb_cpu, self.rb_cuda, self.rb_hip]:
            self.device_group.addButton(btn)


class SettingsController:
    """
    Settings UI와 Core를 연결하는 클래스입니다.
    """
    def __init__(self, core: SettingsCore, ui: SettingsUI):
        self.core = core
        self.ui = ui
        self._connect_signals()

    def _connect_signals(self):
        self.ui.device_group.buttonClicked.connect(self.on_device_radio_clicked)

    def _update_device_setting(self, btn):
        device_type = None
        if btn == self.ui.rb_cpu:
            device_type = "cpu"
        elif btn == self.ui.rb_cuda:
            device_type = "cuda"
        elif btn == self.ui.rb_hip:
            device_type = "hip"
        # DB update
        training = self.core.load_settings().get('training', {})
        training['device'] = device_type
