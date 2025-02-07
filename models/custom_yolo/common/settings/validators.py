
class SettingValidator:
    _valid_rules = {
        'training.batch_size': lambda v: 1 <= v <= 256,
        'training.epochs': lambda v: 1 <= v <= 100,
        'training.device': lambda v: v in ['cpu', 'cuda', 'hip'],
    }

    @classmethod
    def validate_settings(cls, key_path, value):
        if key_path in cls._valid_rules:
            if not cls._valid_rules[key_path](value):
                raise ValueError(f"Invalid value for {key_path}: {value}")