PRAGMA foreign_keys = ON;  -- SQLite에서 FK 제약조건 활성화

BEGIN TRANSACTION;

/*
 * 이미 작성하셨던 기존 테이블들
 */
CREATE TABLE document_converter
(
    id            INTEGER PRIMARY KEY,
    poppler_path  TEXT    DEFAULT '/usr/bin/poppler',
    output_dir    TEXT    DEFAULT './output'
);

CREATE TABLE global_settings
(
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

/*
 * training_config에 YOLO_model, custom_model_path, data_yaml
 * 등을 추가로 저장할 수 있도록 칼럼을 확장
 */
CREATE TABLE training_config
(
    id              INTEGER PRIMARY KEY,
    epochs          INTEGER DEFAULT 50,
    batch_size      INTEGER DEFAULT 16,
    device          TEXT    DEFAULT 'cpu'
                       CHECK (device IN ('cpu', 'cuda', 'hip')),
    imgsz           INTEGER DEFAULT 640,
    YOLO_model      TEXT,   -- ex) "C:/.../yolo11n.pt"
    custom_model_path TEXT, -- ex) "custom.pt"
    data_yaml       TEXT    -- ex) "C:/.../custom_data.yaml"
);

/*
 * 1) class_info:
 *    "0": "question", "1": "answer" 처럼
 *    클래스 인덱스와 클래스 이름을 매핑
 */
CREATE TABLE class_info
(
    class_idx   INTEGER PRIMARY KEY,
    class_name  TEXT NOT NULL
);

/*
 * 2) labeling_config:
 *    labeling 관련 JSON("labeling": {...})을 저장
 */
CREATE TABLE labeling_config
(
    id                  INTEGER PRIMARY KEY,
    device              TEXT    DEFAULT 'cpu'
                           CHECK (device IN ('cpu', 'cuda', 'hip')),
    label_save_mode     TEXT    DEFAULT 'manual', -- ex) 'manual' or 'auto'
    llm_max_retries     INTEGER DEFAULT 5,
    detect_imgsz        INTEGER DEFAULT 640,
    detect_conf         REAL    DEFAULT 0.5,      -- 0.0 ~ 1.0
    llm_max_new_tokens  INTEGER DEFAULT 10000
);

/*
 * 3) other_settings:
 *    "use_gpu", "log_level" 등
 *    단순 key-value를 넘어 세부 필드가 필요한 경우 테이블 분리
 */
CREATE TABLE other_settings
(
    id          INTEGER PRIMARY KEY,
    use_gpu     INTEGER DEFAULT 0,  -- 0=false, 1=true
    log_level   TEXT    DEFAULT 'INFO'
);

/*
 * 4) data_config:
 *    "data": { "path": "...", "train": "...", "val": "..." } 를 저장
 */
CREATE TABLE data_config
(
    id          INTEGER PRIMARY KEY,
    base_path   TEXT,  -- data.path
    train_path  TEXT,  -- data.train
    val_path    TEXT   -- data.val
);

/*
 * 5) results_config:
 *    "results": {"path": "...", "train": "...", "val": "..."} 를 저장
 */
CREATE TABLE results_config
(
    id           INTEGER PRIMARY KEY,
    base_path    TEXT,  -- results.path
    train_path   TEXT,  -- results.train
    val_path     TEXT   -- results.val
);

/*
 * 6) model_paths:
 *    "detector_model_path", "llm_model" 등을 저장
 */
CREATE TABLE model_paths
(
    id                    INTEGER PRIMARY KEY,
    detector_model_path   TEXT,  -- ex) "C:/.../best.pt"
    llm_model             TEXT   -- ex) "microsoft/phi-2"
);

/*
 * 예: training_config가 data_config, results_config, model_paths를 참조하게끔
 *    (원할 경우 외래키 관계를 추가)
 */
ALTER TABLE training_config ADD COLUMN data_config_id INTEGER REFERENCES data_config(id);
ALTER TABLE training_config ADD COLUMN results_config_id INTEGER REFERENCES results_config(id);
ALTER TABLE training_config ADD COLUMN model_paths_id INTEGER REFERENCES model_paths(id);

COMMIT;
