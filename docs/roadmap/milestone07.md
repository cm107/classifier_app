# Milestone 7: Microsoft Dataset Integration & Validation

### Goal:
Programmatically integrate the Microsoft Cats vs. Dogs dataset and execute a benchmark training run to verify the robustness of the existing data pipeline (Milestones 1–6).

---

## 7.1: Automated Data Ingestion
**Goal:** Use `kagglehub` to fetch the specific Microsoft version of the dataset.

* [ ] **Update `KaggleProvider` Logic**:
    * Configure the provider to target `shaunthesheep/microsoft-catsvsdogs-dataset`.
    * **Implementation**:
      ```python
      path = kagglehub.dataset_download('shaunthesheep/microsoft-catsvsdogs-dataset')
      ```
* [ ] **Path Management**:
    * Map the download path to your internal `storage/datasets/microsoft_cats_dogs/` directory.

## 7.2: Data Cleaning (The "Pruning" Phase)
**Goal:** This specific dataset is known to contain a few corrupted images and non-image files (like `Thumbs.db`) that can crash a PyTorch training loop.

* [ ] **File Sanitization**:
    * Implement a script to scan the `PetImages/` folder.
    * [ ] Remove non-JPG files (e.g., `.db`, `.txt`).
    * [ ] **Corruption Check**: Attempt to open each image with `PIL.Image`. Delete files that throw an `UnidentifiedImageError`.
* [ ] **Directory Alignment**:
    * Ensure the structure is:
      - `PetImages/Cat/`
      - `PetImages/Dog/`

## 7.3: Metadata Indexing
**Goal:** Register the new dataset with your Milestone 1 Physical Layer.

* [ ] **Generate Manifest**:
    * Run `MetadataManager.generate_metadata()` on the cleaned `PetImages` directory.
    * Verify that the class labels are correctly inferred from the folder names (`Cat`, `Dog`).
* [ ] **Split Validation**:
    * Verify a consistent sample count (approx. 12,500 per class before pruning).

## 7.4: Benchmark Training Run
**Goal:** Verify that the Milestone 4 Training Engine and Milestone 5 Model Architecture work with real-world data.

* [ ] **Execution**:
    * Load the dataset via your `ClassificationDataset` class.
    * Run a 3-epoch "Smoke Test."
* [ ] **Success Criteria**:
    * Loss decreases consistently over 3 epochs.
    * The system successfully saves a `checkpoint.pth` at the end of the run.

---

### Comparison of Workflows
| Task | Previous (Competition Version) | Updated (Microsoft Version) |
| :--- | :--- | :--- |
| **Download** | `competition_download` | `dataset_download` |
| **Organization** | Required Regex (flat files) | **Not Required** (already in folders) |
| **Cleaning** | Standard validation | **Mandatory** (needs corruption check) |
