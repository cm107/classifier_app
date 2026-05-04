# Logging System: Design and Implementation Patterns

## 1. Overview
The logging system provides a centralized "Source of Truth" for the application's runtime behavior. By using a custom **Logger class**, we maintain consistency across the **MVC/MVVM architecture**, allowing for precise debugging of asynchronous tasks like model training and live video streaming.

## 2. Global Accessibility & Singleton Instance
To ensure the logger is accessible across all modules without circular imports, it is instantiated in its own script.

*   **Instance Location**: `vision_app/core/logger.py`
*   **Usage**: 
    ```python
    from vision_app.core.logger import log

    # Standard usage:
    log.info("DatasetBuilder", "Successfully initialized metadata manager.")
    ```

## 3. Log Entry Format
Every log entry requires a **Source Identifier** as the first parameter to identify exactly where the log was triggered. This is critical for tracing logic flow through submodule interactions.

*   **Format**: `log.level("SourceIdentifier", "Message")`
*   **Example**: `log.debug("ResidualBlock", "He Initialization applied to conv weights.")`.

## 4. Log Levels and Routing
Logs are routed based on severity and the application's configuration settings in `config.yaml`.

| Level | Routing | Description |
| :--- | :--- | :--- |
| **DEBUG** | Logs only | **Temporary** entries for active problem-solving. These must be removed after the specific issue is resolved. |
| **VERBOSE** | Logs only | **Permanent** but granular. Gated by specific flags in `config.yaml` (e.g., `verbose_training: True`). |
| **INFO** | Logs + Stdout | Highlights **core events** (e.g., "Starting Milestone 7 data ingestion"). |
| **WARNING** | Logs + Stdout | Information requiring **user attention** that does not halt execution (e.g., "Dataset imbalance detected"). |
| **ERROR** | Logs + Stdout | Indicates a **fatal problem** followed by termination. Does **not** include a stack trace. |
| **EXCEPTION** | Logs + Stdout | Indicates an **unexpected failure**. Automatically includes a full stack trace using the `traceback` library. |

## 5. Standard Practice: Exception Handling
To prevent the PySide6 UI from freezing or crashing, all critical logic should be encapsulated in **try-except blocks**, with the `EXCEPTION` level used in catch blocks.

```python
try:
    # Logic for file system operations or model forward passes
    self.handler.move_sample(src, dest)
except Exception as e:
    log.exception("FileSystemHandler", f"Failed to move sample: {e}")
```

## 6. Storage and Persistent Sessions
*   **Directory**: All logs are saved to the `storage/logs/` directory.
*   **Initialization**: The `main.py` entry point ensures this directory exists at startup.
*   **Session-Based Naming**: To distinguish between different application runs, every session generates a unique log file name including the date and the specific time down to the second.
    *   **Format**: `app_YYYY-MM-DD_HH-mm-ss.log`
    *   **Example**: `app_2026-05-04_14-35-02.log`

By incorporating the session time into the filename, developers can easily correlate specific log files with recorded training metrics or UI performance benchmarks.