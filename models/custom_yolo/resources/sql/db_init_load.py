class DB_INIT_QUERY:

    def __init__(self):
        self.query = """
        PRAGMA foreign_keys = ON;
        
        BEGIN TRANSACTION;
        
        -- 테이블 존재 시 삭제(주의: 실제 운영에서는 신중하게 사용)
        DROP TABLE IF EXISTS training_config;
        DROP TABLE IF EXISTS labeling_config;
        DROP TABLE IF EXISTS other_settings;
        DROP TABLE IF EXISTS data_config;
        DROP TABLE IF EXISTS results_config;
        DROP TABLE IF EXISTS model_paths;
        DROP TABLE IF EXISTS class_info;
        DROP TABLE IF EXISTS document_converter;
        DROP TABLE IF EXISTS global_settings;
        
        -- 1) document_converter
        CREATE TABLE document_converter
        (
            id            INTEGER PRIMARY KEY,
            poppler_path  TEXT    DEFAULT '/usr/bin/poppler',
            output_dir    TEXT    DEFAULT './output'
        );
        
        -- 2) global_settings
        CREATE TABLE global_settings
        (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        
        -- 3) class_info
        CREATE TABLE class_info
        (
            class_idx   INTEGER PRIMARY KEY,
            class_name  TEXT NOT NULL
        );
        
        -- 4) data_config
        CREATE TABLE data_config
        (
            id          INTEGER PRIMARY KEY,
            base_path   TEXT,
            train_path  TEXT,
            val_path    TEXT
        );
        
        -- 5) results_config
        CREATE TABLE results_config
        (
            id           INTEGER PRIMARY KEY,
            base_path    TEXT,
            train_path   TEXT,
            val_path     TEXT
        );
        
        -- 6) model_paths
        CREATE TABLE model_paths
        (
            id                    INTEGER PRIMARY KEY,
            detector_model_path   TEXT,
            llm_model             TEXT
        );
        
        -- 7) labeling_config
        CREATE TABLE labeling_config
        (
            id                  INTEGER PRIMARY KEY,
            device              TEXT    DEFAULT 'cpu'
                                   CHECK (device IN ('cpu','cuda','hip')),
            label_save_mode     TEXT    DEFAULT 'manual',
            llm_max_retries     INTEGER DEFAULT 5,
            detect_imgsz        INTEGER DEFAULT 640,
            detect_conf         REAL    DEFAULT 0.5,
            llm_max_new_tokens  INTEGER DEFAULT 10000
        );
        
        -- 8) other_settings
        CREATE TABLE other_settings
        (
            id          INTEGER PRIMARY KEY,
            use_gpu     INTEGER DEFAULT 0,  -- 0=false, 1=true
            log_level   TEXT    DEFAULT 'INFO'
        );
        
        -- 9) training_config
        CREATE TABLE training_config
        (
            id              INTEGER PRIMARY KEY,
            epochs          INTEGER DEFAULT 50,
            batch_size      INTEGER DEFAULT 16,
            device          TEXT    DEFAULT 'cpu'
                               CHECK (device IN ('cpu','cuda','hip')),
            imgsz           INTEGER DEFAULT 640,
            YOLO_model      TEXT,
            custom_model_path TEXT,
            data_yaml       TEXT,
        
            data_config_id    INTEGER REFERENCES data_config(id),
            results_config_id INTEGER REFERENCES results_config(id),
            model_paths_id    INTEGER REFERENCES model_paths(id)
        );
        
        COMMIT;
        """

    def get_init_query(self):
        return self.query
