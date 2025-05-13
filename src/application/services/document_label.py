import os
import io
tempfile
import cv2
from fastapi import UploadFile
from typing import List, Dict
from models.custom_yolo.infrastructure.computer_vision.label_model.auto_labeler import AutoLabeler

class DocumentLabelService:
    def __init__(self):
        # Initialize AutoLabeler with default settings
        self.auto_labeler = AutoLabeler()
        model_path = os.getenv("DETECTOR_MODEL_PATH")
        self.auto_labeler.initialize_detector(model_path)

    async def label(self, upload_file: UploadFile) -> List[Dict]:
        """
        Detect objects and generate labels for an uploaded document (image).
        Returns a list of label info dicts.
        """
        content = await upload_file.read()
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            boxes = self.auto_labeler.detect_objects(tmp_path)
            img = cv2.imread(tmp_path)
            labels = []
            for box in boxes:
                info = self.auto_labeler.generate_label_for_box(img, box)
                labels.append(info)
            return labels
        finally:
            os.remove(tmp_path)
