from dataclasses import dataclass
from typing import List
from models.custom_yolo.core.services.crud_helper import TableSchema


DEVICE_OPTIONS = ("cpu", "cuda", "hip")


@dataclass
class LabelingConfigSchema(TableSchema):
    name: str = "labeling_config"
    pk: str = "id"
    columns: List[str] = (
        "id", "device", "label_save_mode", "llm_max_retries",
        "detect_imgsz", "detect_conf", "llm_max_new_tokens"
    )


@dataclass
class TrainingConfigSchema(TableSchema):
    name: str = "training_config"
    pk: str = "id"
    columns: List[str] = (
        "id", "epochs", "batch_size", "device", "imgsz",
        "YOLO_model", "custom_model_path", "data_yaml",
        "data_config_id", "results_config_id", "model_paths_id"
    )


@dataclass
class DocumentConverterSchema(TableSchema):
    name: str = "document_converter"
    pk: str = "id"
    columns: List[str] = (
        "id", "poppler_path", "output_dir", "dpi",
        "pdf_path", "output_format", "thread_count", "open_after_finish"
    )


@dataclass
class GlobalSettingsSchema(TableSchema):
    name: str = "global_settings"
    pk: str = "key"
    columns: list = ("key", "value")


@dataclass
class ClassInfoSchema(TableSchema):
    name: str = "class_info"
    pk: str = "class_idx"
    columns: List[str] = ("class_idx", "class_name")


@dataclass
class DataConfigSchema(TableSchema):
    name: str = "data_config"
    pk: str = "id"
    columns: List[str] = ("id", "base_path", "train_path", "val_path")


@dataclass
class ModelPathsSchema(TableSchema):
    name: str = "model_paths"
    pk: str = "id"
    columns: List[str] = ("id", "detector_model_path", "llm_model")


@dataclass
class OtherSettingsSchema(TableSchema):
    name: str = "other_settings"
    pk: str = "id"
    columns: List[str] = ("id", "use_gpu", "log_level")


@dataclass
class ResultsConfigSchema(TableSchema):
    name: str = "results_config"
    pk: str = "id"
    columns: List[str] = ("id", "base_path", "train_path", "val_path")

