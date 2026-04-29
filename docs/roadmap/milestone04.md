Milestone 4 is where your application starts to look like a professional tool. This phase focuses on the **Main Window** (the shell) and the **Dataset Management** tab. Because your `data_manager.py` logic is already built and tested, the UI's job is simply to trigger those methods and display the results.

Here is the implementation TODO list for **Step 1.4: The GUI Shell & Dataset UI**.

---

### 1.4.1: Main Window Implementation (`main_window.py`)
*Goal: Create a stable, navigable "Home" for all your sub-widgets.*

* [ ] **`class NavigationController`**:
    * Implement a `QListWidget` or `QToolBar` acting as a sidebar with icons for "Data," "Train," "Models," and "Inference."
    * Connect the sidebar's `currentItemChanged` signal to a `QStackedWidget` to swap between the four main tabs.
* [ ] **`class StatusBarManager`**:
    * Implement a permanent `QLabel` showing the current GPU (e.g., "NVIDIA RTX 3060") using `torch.cuda.get_device_name()`.
    * Add a temporary message area for notifications like "Dataset 'Apples' successfully split."
* [ ] **`class ThemeEngine`**:
    * Create a `style.qss` file in `resources/` (Dark Mode is usually preferred for AI tools).
    * Implement a method to apply this stylesheet to the `QApplication` instance.

---

### 1.4.2: Dataset Management Widget (`dataset_tab.py`)
*Goal: Provide a visual interface for the `DatasetBuilder` submodules.*

* [ ] **`class DatasetManagerWidget(QWidget)`**:
    * **The List View**: Implement a `QTreeWidget` to show the folders in `storage/datasets/`. Each folder should expand to show `train`, `val`, and `test` sub-counts.
    * **Action Buttons**: Add "New Dataset," "Import Images," "Merge," and "Split" buttons.
    * **Logic Wiring**: Connect the "Split" button to a `QInputDialog` (for the ratio) and then to the `DatasetTransformer.split_train_val_test()` method.
* [ ] **`class PruningDialog(QDialog)`**:
    * Create a popup with checkboxes for "Remove Duplicates" and "Verify Corruptions."
    * Implement a `QProgressBar` that reflects the progress of the `DatasetTransformer.prune_dataset()` operation.

---

### 1.4.3: Entry Point Logic (`main.py`)
*Goal: Initialize the environment and launch the app.*

* [ ] **The `main()` Function**:
    * Set up High-DPI scaling attributes (`Qt.AA_EnableHighDpiScaling`).
    * Instantiate `QApplication`.
    * Check for `storage/` directory existence; create them if they are missing (folders for `datasets`, `models`, `logs`).
    * Show `MainWindow` and execute the app loop.

---

### Verification Checklist for 1.4
1.  **Tab Navigation**: Clicking each sidebar icon should switch the central view instantly without flickering or crashing.
2.  **Live Updates**: When you "Import" a folder of images via the UI, the `QTreeWidget` should refresh to show the new file count immediately.
3.  **UI Responsiveness**: Ensure that when a `QDialog` is open, the rest of the `MainWindow` remains stable (modal behavior).
4.  **Path Resolution**: Verify that the app correctly identifies the `storage/` path relative to `main.py` regardless of where the terminal is launched.

**Would you like me to move on to detailing the TODO list for Milestone 5 (Training Monitor & Hyperparameters)?**