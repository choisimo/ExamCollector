BEGIN TRANSACTION;

/*
 * model_paths (detector_model_path, llm_model)
 * JSON:
 *   "detector_model_path": "C:/WorkSpace/git/ExamCollector/runs/detect/train12/weights/best.pt",
 *   "llm_model": "microsoft/phi-2"
 */
INSERT INTO model_paths (id, detector_model_path, llm_model)
VALUES (
  1,
  'C:/WorkSpace/git/ExamCollector/runs/detect/train12/weights/best.pt',
  'microsoft/phi-2'
);


/*
 * data_config (base_path, train_path, val_path)
 * JSON:
 *   "data": {
 *     "path": "C:/WorkSpace/git/ExamCollector/models/custom_yolo/data/test-2",
 *     "train": "train/images",
 *     "val": "valid/images"
 *   }
 */
INSERT INTO data_config (id, base_path, train_path, val_path)
VALUES (
  1,
  'C:/WorkSpace/git/ExamCollector/models/custom_yolo/data/test-2',
  'train/images',
  'valid/images'
);


/*
 * results_config (base_path, train_path, val_path)
 * JSON:
 *   "results": {
 *     "path": "C:/WorkSpace/git/ExamCollector/models/custom_yolo/model/label_model/results",
 *     "train": "train",
 *     "val": "val"
 *   }
 */
INSERT INTO results_config (id, base_path, train_path, val_path)
VALUES (
  1,
  'C:/WorkSpace/git/ExamCollector/models/custom_yolo/model/label_model/results',
  'train',
  'val'
);


/*
 * class_info (class_idx, class_name)
 * JSON:
 *   "class_info": {
 *     "0": "question",
 *     "1": "answer",
 *     "2": "figure",
 *     "3": "etc",
 *     "4": "q_num",
 *     "5": "q_type"
 *   }
 */
INSERT INTO class_info (class_idx, class_name) VALUES (0, 'question');
INSERT INTO class_info (class_idx, class_name) VALUES (1, 'answer');
INSERT INTO class_info (class_idx, class_name) VALUES (2, 'figure');
INSERT INTO class_info (class_idx, class_name) VALUES (3, 'etc');
INSERT INTO class_info (class_idx, class_name) VALUES (4, 'q_num');
INSERT INTO class_info (class_idx, class_name) VALUES (5, 'q_type');


/*
 * training_config
 * JSON ("training"): {
 *   "epochs": 100,
 *   "batch_size": 50,
 *   "device": "cuda",
 *   "imgsz": 640,
 *   "YOLO_model": "...",
 *   "custom_model_path": "...",
 *   "data_yaml": "..."
 * }
 *  + 연결될 FK:
 *    data_config_id    = 1
 *    results_config_id = 1
 *    model_paths_id    = 1
 */
INSERT INTO training_config (
    id, epochs, batch_size, device, imgsz,
    YOLO_model, custom_model_path, data_yaml,
    data_config_id, results_config_id, model_paths_id
)
VALUES (
  1,
  100,
  50,
  'cuda',
  640,
  'C:/WorkSpace/git/ExamCollector/models/custom_yolo/model/label_model/yolo11n.pt',
  'custom.pt',
  'C:/WorkSpace/git/ExamCollector/models/custom_yolo/core/use_cases/custom_data.yaml',
  1,
  1,
  1
);


/*
 * labeling_config
 * JSON ("labeling"): {
 *   "device": "cpu",
 *   "label_save_mode": "manual",
 *   "llm_max_retries": 5,
 *   "detect_imgsz": 640,
 *   "detect_conf": 0.5,
 *   "llm_max_new_tokens": 10000
 * }
 */
INSERT INTO labeling_config (
    id, device, label_save_mode, llm_max_retries,
    detect_imgsz, detect_conf, llm_max_new_tokens
)
VALUES (
  1,
  'cpu',
  'manual',
  5,
  640,
  0.5,
  10000
);


/*
 * other_settings
 * JSON ("other_settings"): {
 *   "use_gpu": true,
 *   "log_level": "INFO"
 * }
 * use_gpu = true -> 1
 */
INSERT INTO other_settings (id, use_gpu, log_level)
VALUES (
  1,
  1,
  'INFO'
);


/*
 * document_converter
 * JSON ("document_converter"): {
 *   "poppler_path": "C:/poppler-24.08.0/Library/bin",
 *   "output_dir": "C:/"
 * }
 */
INSERT INTO document_converter (id, poppler_path, output_dir)
VALUES (
  1,
  'C:/poppler-24.08.0/Library/bin',
  'C:/'
);


/*
 * global_settings
 * JSON ("settings_path"): "C:/WorkSpace/git/ExamCollector/models/custom_yolo/core/domain/settings.json"
 *   -> key/value
 */
INSERT INTO global_settings (key, value)
VALUES (
  'settings_path',
  'C:/WorkSpace/git/ExamCollector/models/custom_yolo/core/domain/settings.json'
);

COMMIT;
