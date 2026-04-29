# Data & Path Management Rules

## 1. Directory Standards: The "ImageFolder" Hierarchy
To ensure compatibility with PyTorch's `ImageFolder` and the application's internal logic, all datasets must follow a strict physical hierarchy.

- **Structure**: `storage/datasets/[dataset_name]/[split]/[class_name]/[image_file]`
- **Splits**: Every dataset must be divided into `train/` and `val/` subdirectories (with an optional `test/` subdirectory).
- **Class Labels**: Folder names under each split represent the class strings (e.g., `Apple`, `Orange`). These must be consistent across all splits.
- **80/20 Rule**: Default logic for new datasets should implement an 80% training and 20% validation split.

## 2. Centralized Storage Root
All application data must be contained within a single `storage/` directory located at the project root.
- **`storage/datasets/`**: For all raw and processed image data.
- **`storage/models/`**: For saving weights (`.pth`) and exported TorchScript models (`.pt`).
- **`storage/logs/`**: For training metrics and system logs.
- **`storage/trash/`**: For images flagged by the **Corrupt Filter** or **Duplicate Detection** logic. Never delete user data permanently; move it here instead.

## 3. Path Resolution & Portability
- **Library**: Always use the `pathlib` library for path manipulations to ensure cross-platform compatibility (Windows/Linux).
- **Relative Pathing**: All paths must be resolved relative to the location of `main.py`. Do not use hardcoded absolute paths.
- **Initialization**: At application startup (`main.py`), verify that the `storage/` root and all required subdirectories exist. Create them if they are missing.

## 4. Metadata: The "Source of Truth"
The application must not rely on constant disk scanning.
- **Metadata File**: Every dataset directory must contain a `metadata.json` or `dataset_info.json`.
- **Synchronization**: Any operation that modifies the disk (e.g., importing images, pruning duplicates, or splitting a dataset) **must** immediately trigger an update to the metadata file.
- **Contents**: The metadata should track class counts, label mappings (int to string), and creation dates.

## 5. Safe File System Operations
When generating logic for the `FileSystemHandler` or `DatasetTransformer` submodules:
- **Collision Prevention**: When moving or importing files, check if a filename already exists in the destination. If so, append a suffix (e.g., `image_1.jpg`) instead of overwriting.
- **Cleanliness**: Empty class folders should be automatically identified and removed to keep the "ImageFolder" structure lean.
- **Verification**: Before any file operation, verify that the source path exists and the destination is writable.

## 6. Global Configuration
- **`config.yaml`**: Global paths (e.g., `dataset_root`, `model_export_root`) and hardware settings must be read from the central YAML configuration.
- **No Hardcoding**: Never hardcode directory strings directly into the `core` or `ui` logic. Always use the `ConfigLoader` utility.
