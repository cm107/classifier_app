# Logging Rules & Best Practices

## 1. Centralized Logger Singleton
To maintain a consistent "Source of Truth" for application events, all logging must be performed through the centralized `Logger` instance.

- **Import Path**: Always use `from vision_app.core.logger import log`.
- **Initialization**: Do not instantiate new loggers in local files. The singleton instance is managed in `vision_app/core/logger.py` and is accessible globally.

## 2. Mandatory Log Entry Structure
Every log call must include a **Source Identifier** as the first parameter. This identifier should be the name of the Class or Function where the log is generated to ensure logic flow can be traced through the MVC/MVVM layers.

- **Syntax**: `log.level("SourceIdentifier", "Message")`
- **Example**: `log.info("ModelTrainer", "Starting supervised fine-tuning phase.")`

## 3. Log Levels & Routing
Follow these specific rules for selecting the appropriate log level:

- **DEBUG**: Use for **temporary** debugging of specific logic issues. These must be cleaned up/removed after the problem is resolved. (Log file only).
- **VERBOSE**: Use for **permanent** but high-granularity technical details. These must be gated by a flag in the settings (e.g., `config.yaml`). (Log file only).
- **INFO**: Use for **core application events** (e.g., dataset ingestion, milestone completion). (Log file + Stdout).
- **WARNING**: Use for events requiring **user attention** that do not indicate a failure (e.g., unexpected data formats that are still processable). (Log file + Stdout).
- **ERROR**: Use for **fatal logic failures** that lead to application termination. Use sparingly and do not include a stack trace. (Log file + Stdout).
- **EXCEPTION**: Use for **unexpected runtime failures**. This method must automatically include a full stack trace using the `traceback` library. (Log file + Stdout).

## 4. Error Handling Pattern
To ensure the PySide6 UI remains stable and does not crash unexpectedly, encapsulate critical logic (File I/O, AI Math, Threading) in `try-except` blocks.

- **Requirement**: Catch blocks should prioritize `log.exception` to ensure the diagnostic stack trace is captured.
```python
try:
    self.trainer.start_training()
except Exception as e:
    log.exception("TrainWorker", f"Training failed unexpectedly: {e}")
```

## 5. Storage & Session Management
- **Directory**: Ensure all logs are directed to `storage/logs/` as defined in the configuration.
- **Session Naming**: Log files must use a unique timestamp for every session to include the specific time down to the second.
- **Format**: `app_YYYY-MM-DD_HH-mm-ss.log` (e.g., `app_2026-05-04_14-35-02.log`).

## 6. UI Synchronization
- All messages destined for the `StatusBarManager` or `status_changed` signals in `TrainWorker` or `StreamWorker` should also be mirrored as `log.info` or `log.warning` entries to ensure the persistent log file matches the user's visual experience.
