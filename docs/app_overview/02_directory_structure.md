This is an ambitious and highly structured project. Given your requirements, a **Model-View-Controller (MVC)** or **Model-View-ViewModel (MVVM)** approach is best to keep the heavy PyTorch logic (the "Engine") separate from the PySide6 UI code.

Here is the proposed directory structure for your project, titled `core_vision_app`.

```text
/vision_project
│
├── main.py                     # Entry point (initializes the QApplication)
├── config.yaml                 # Global app settings (default paths, UI themes)
│
└── vision_app/                 # Main Python Package
    ├── __init__.py
    │
    ├── core/                   # The "Engine" (Pure PyTorch & Logic)
    │   ├── __init__.py
    │   ├── model.py            # ScratchResNet and ResidualBlock definitions
    │   ├── trainer.py          # Training loops, SSL logic, and NT-Xent loss
    │   ├── dataset.py          # Custom Dataset classes and Augmentation logic
    │   ├── data_manager.py     # Pruning, splitting, and combining logic
    │   ├── inference.py        # Inference engine (Image, Video, Webcam)
    │   └── utils.py            # Stats calculation (Mean/Std), TorchScript export
    │
    ├── ui/                     # PySide6 Layouts & Components
    │   ├── __init__.py
    │   ├── main_window.py      # The primary shell with navigation sidebar
    │   ├── widgets/            # Reusable UI components
    │   │   ├── dataset_tab.py  # UI for creating/pruning datasets
    │   │   ├── hyper_param.py  # UI for editing config/hyperparameters
    │   │   ├── monitor.py      # Real-time progress bars and loss plots
    │   │   ├── model_list.py   # UI for managing/deleting/finetuning weights
    │   │   └── live_view.py    # UI for Webcam/Video inference display
    │   └── resources/          # Icons, Stylesheets (QSS), and Images
    │
    ├── worker/                 # Multithreading (QThreads)
    │   ├── __init__.py
    │   ├── train_worker.py     # Background thread for non-blocking training
    │   └── stream_worker.py    # Background thread for webcam/video processing
    │
    └── storage/                # Local data management
        ├── datasets/           # Default root for user-created datasets
        ├── models/             # Saved .pth and .pt (TorchScript) files
        └── logs/               # Training logs (CSV/TensorBoard)
```

---

### Key Structural Decisions:
* **`vision_app/core`**: This folder should be able to run as a standalone CLI if needed. It contains zero PySide6 code, ensuring your AI logic isn't "entangled" with the UI.
* **`vision_app/worker`**: Crucial for PySide6. Training a model or running a webcam on the main thread will "freeze" the UI. These files will handle `QThread` communication via signals.
* **`vision_app/ui/widgets`**: Breaking the tabs into separate files makes the project manageable. Each functionality (Dataset Management, Hyperparameters, etc.) is isolated.

---

**Would you like me to dive into the details of `vision_app/core/data_manager.py` to define how the dataset pruning and combining logic will work?**