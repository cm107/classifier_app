To get Milestone 1 off the ground, we need to build the "Physical Layer" of your data management. These submodules in `data_manager.py` will handle the actual files on your disk, ensuring that your PyTorch `Dataset` always has a clean, predictable environment to read from.

Here is the implementation TODO list for **Step 1.1: Data Management Implementation**.

---

### 1.1.1: FileSystemHandler Submodule
*Goal: Abstract all `os` and `shutil` operations into a safe, high-level API.*

* [ ] **Initialize with Parent**: Store `self._parent` to access global app paths (e.g., `storage/datasets/`).
* [ ] **`create_dataset_structure(name, classes)`**: Create a root folder with `train`, `val`, and `test` subdirectories, each containing a folder for each class string.
* [ ] **`move_sample(src_path, dest_folder)`**: Move a single image file while ensuring no filename collisions (e.g., appending `_1` if the name exists).
* [ ] **`get_class_distribution(root_path)`**: Scan a dataset and return a dictionary of `{class_name: file_count}`.
* [ ] **`delete_empty_classes(dataset_path)`**: Clean up the directory tree by removing any class folders that contain zero images.

### 1.1.2: MetadataManager Submodule
*Goal: Maintain a "Source of Truth" JSON file so the app doesn't have to scan the disk constantly.*

* [ ] **`generate_metadata(dataset_path)`**: Create a `metadata.json` containing the creation date, number of classes, and total image count.
* [ ] **`update_stats(dataset_path)`**: A method to refresh the JSON whenever the user adds or prunes images.
* [ ] **`load_metadata(dataset_path)`**: Read the JSON and return a structured object for the UI to display in the "Dataset Tab."
* [ ] **`get_label_map(dataset_path)`**: Create a mapping of `int` to `string` (e.g., `0: "Apple", 1: "Orange"`) required for the final classification layer.

### 1.1.3: DatasetTransformer Submodule
*Goal: Perform heavy-duty structural changes like splitting and merging.*

* [ ] **`split_train_val_test(root_path, ratios=(0.8, 0.1, 0.1))`**: 
    * Iterate through "Raw" folders.
    * Shuffle filenames with a fixed seed (for reproducibility).
    * Move files into their respective `train`/`val`/`test` slots based on percentages.
* [ ] **`merge_datasets(source_paths, target_path)`**: Combine multiple datasets into one, handling cases where class names overlap or differ.
* [ ] **`prune_dataset(dataset_path, criteria)`**: 
    * Implement **Duplicate Detection**: Use hash-based checking (MD5/SHA) to remove identical files.
    * Implement **Corrupt Filter**: Attempt to open every file with `PIL.Image`; move failures to a "Trash" folder.

---

### 1.1.4: Dataset Integration (dataset.py)
*Goal: Connect the physical folders to the PyTorch training loop.*

* [ ] **`ClassificationDataset(Dataset)`**:
    * Inherit from `torch.utils.data.Dataset`.
    * Implement `__len__` and `__getitem__`.
    * Integrate a "Label Smoothing" flag to decide whether to return hard labels or soft-encoded vectors.
* [ ] **`ContrastiveTransformations`**:
    * Implement the `__call__` method to return a tuple `(view_1, view_2)` from a single input image.
    * Hardcode the augmentation pipeline (RandomResizedCrop, ColorJitter, Blur).

---

### Verification Checklist for 1.1
Before moving to Milestone 2, you should be able to run a script that does the following:
1.  **Create** a dataset named "TestProject" with classes "A" and "B".
2.  **Verify** `metadata.json` exists and correctly lists 2 classes.
3.  **Split** the data and confirm the file counts in `train/` and `val/` match your 80/20 request.
4.  **Load** one sample into `ClassificationDataset` and confirm it is a valid `torch.Tensor`.

**Would you like me to move on to detailing the TODO list for Milestone 2 (The Model & Training Engine)?**