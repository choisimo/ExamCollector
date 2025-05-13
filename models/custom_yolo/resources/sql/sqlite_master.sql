INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'document_converter', 'document_converter', 2, 'CREATE TABLE document_converter
(
    id            INTEGER PRIMARY KEY,
    poppler_path  TEXT    DEFAULT ''/usr/bin/poppler'',
    output_dir    TEXT    DEFAULT ''./output''
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'global_settings', 'global_settings', 3, 'CREATE TABLE global_settings
(
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('index', 'sqlite_autoindex_global_settings_1', 'global_settings', 4, null);
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'training_config', 'training_config', 5, 'CREATE TABLE training_config
(
    id              INTEGER PRIMARY KEY,
    epochs          INTEGER DEFAULT 50,
    batch_size      INTEGER DEFAULT 16,
    device          TEXT    DEFAULT ''cpu''
                       CHECK (device IN (''cpu'', ''cuda'', ''hip'')),
    imgsz           INTEGER DEFAULT 640,
    YOLO_model      TEXT,   -- ex) "C:/.../yolo11n.pt"
    custom_model_path TEXT, -- ex) "custom.pt"
    data_yaml       TEXT    -- ex) "C:/.../custom_data.yaml"
, data_config_id INTEGER REFERENCES data_config(id), results_config_id INTEGER REFERENCES results_config(id), model_paths_id INTEGER REFERENCES model_paths(id))');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'class_info', 'class_info', 6, 'CREATE TABLE class_info
(
    class_idx   INTEGER PRIMARY KEY,
    class_name  TEXT NOT NULL
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'labeling_config', 'labeling_config', 7, 'CREATE TABLE labeling_config
(
    id                  INTEGER PRIMARY KEY,
    device              TEXT    DEFAULT ''cpu''
                           CHECK (device IN (''cpu'', ''cuda'', ''hip'')),
    label_save_mode     TEXT    DEFAULT ''manual'', -- ex) ''manual'' or ''auto''
    llm_max_retries     INTEGER DEFAULT 5,
    detect_imgsz        INTEGER DEFAULT 640,
    detect_conf         REAL    DEFAULT 0.5,      -- 0.0 ~ 1.0
    llm_max_new_tokens  INTEGER DEFAULT 10000
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'other_settings', 'other_settings', 8, 'CREATE TABLE other_settings
(
    id          INTEGER PRIMARY KEY,
    use_gpu     INTEGER DEFAULT 0,  -- 0=false, 1=true
    log_level   TEXT    DEFAULT ''INFO''
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'data_config', 'data_config', 9, 'CREATE TABLE data_config
(
    id          INTEGER PRIMARY KEY,
    base_path   TEXT,  -- data.path
    train_path  TEXT,  -- data.train
    val_path    TEXT   -- data.val
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'results_config', 'results_config', 10, 'CREATE TABLE results_config
(
    id           INTEGER PRIMARY KEY,
    base_path    TEXT,  -- results.path
    train_path   TEXT,  -- results.train
    val_path     TEXT   -- results.val
)');
INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES ('table', 'model_paths', 'model_paths', 11, 'CREATE TABLE model_paths
(
    id                    INTEGER PRIMARY KEY,
    detector_model_path   TEXT,  -- ex) "C:/.../best.pt"
    llm_model             TEXT   -- ex) "microsoft/phi-2"
)');
