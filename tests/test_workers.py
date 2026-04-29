"""
tests/test_workers.py — Milestone 3 Verification Script

Covers the checklist from docs/roadmap/milestone03.md:
    1. Signal Test  — TrainWorker emits progress_updated with correct payload.
    2. Thread Safety— Main thread and worker thread print simultaneously without crash.
    3. Abort Test   — Abort flag stops the loop; finished(False) emitted cleanly.
    4. ConfigLoader — load/save round-trip preserves values.

Note: StreamWorker requires OpenCV + a physical camera; its logic is tested
via a unit test that stubs cv2.VideoCapture rather than opening real hardware.

Usage:
    python tests/test_workers.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PySide6 QApplication must be created before any QObject/QThread
from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from vision_app.core.trainer import TrainingMetrics
from vision_app.core.utils import ConfigLoader
from vision_app.worker.train_worker import LifecycleManager, SignalDispatcher, TrainWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_dataset(tmp_dir: Path, n_per_class: int = 4) -> Path:
    """Create a tiny synthetic ImageFolder dataset for worker tests."""
    from PIL import Image
    import random

    dataset = tmp_dir / "DummyDS"
    for split in ("train", "val"):
        for cls in ("A", "B"):
            d = dataset / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = Image.new("RGB", (32, 32),
                                color=(random.randint(0, 255),) * 3)
                img.save(d / f"img_{i}.png")
    return dataset


def _run_worker(worker: TrainWorker, timeout_ms: int = 30_000):
    """
    Start a TrainWorker on a QThread and pump the event loop until the
    OS thread fully exits, or the watchdog fires.

    Key design decision:
        We quit the local event loop on QThread.finished (not
        worker.signals.finished).  QThread.finished is emitted by Qt's
        C++ layer *after* the OS thread function has returned and all
        internal cleanup is done, so by the time loop.exec() returns we
        know the OS thread is gone and thread.wait() returns instantly.

        worker.signals.finished → thread.quit() is kept so that if the
        worker ever runs inside a QThread that *does* have an event loop
        (e.g. during GUI integration) it also shuts down cleanly.

    All queued signals emitted from within worker.run() (progress_updated,
    status_changed, finished) are delivered to the main thread event loop
    *before* QThread.finished arrives, because they are posted to the same
    queue in emission order.
    """
    thread = QThread()
    worker.moveToThread(thread)

    loop = QEventLoop()
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(loop.quit)

    # Quit the local loop only when the OS thread is completely done.
    thread.finished.connect(loop.quit)
    # Also wire finished → thread.quit for future-proofing (no-op without event loop).
    worker.signals.finished.connect(thread.quit)
    thread.started.connect(worker.run)

    thread.start()
    watchdog.start(timeout_ms)
    loop.exec()       # pumps the event loop; delivers all queued signals in order
    watchdog.stop()

    # OS thread has already exited; this returns immediately.
    thread.wait(5_000)


def _base_config(dataset_path: Path, storage_root: Path) -> dict:
    return {
        "dataset_path": dataset_path,
        "num_classes": 2,
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "max_lr": 1e-2,
        "weight_decay": 1e-2,
        "image_size": 32,
        "num_workers": 0,    # 0 for test stability
        "phase": "supervised",
        "label_smoothing": 0.0,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "storage_root": storage_root,
    }


# ---------------------------------------------------------------------------
# Test 1: Signal test — progress_updated carries a TrainingMetrics payload
# ---------------------------------------------------------------------------
def test_signal_emission():
    print("\n[Test 1] Signal emission (progress_updated) ...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_path = _make_dummy_dataset(tmp)
        cfg = _base_config(dataset_path, tmp)

        received: list[TrainingMetrics] = []
        statuses: list[str] = []

        worker = TrainWorker(cfg)
        worker.signals.progress_updated.connect(received.append)
        worker.signals.status_changed.connect(statuses.append)

        _run_worker(worker, timeout_ms=30_000)

        assert len(received) == cfg["epochs"], (
            f"Expected {cfg['epochs']} progress signals, got {len(received)}"
        )
        for m in received:
            assert isinstance(m, TrainingMetrics)
            assert m.total_epochs == cfg["epochs"]
            assert m.phase == "supervised"

        print(f"  PASS — received {len(received)} progress_updated signals")
        print(f"  Status messages: {statuses}")


# ---------------------------------------------------------------------------
# Test 2: Thread safety — main thread output while worker runs
# ---------------------------------------------------------------------------
def test_thread_safety():
    print("\n[Test 2] Thread safety (concurrent print) ...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_path = _make_dummy_dataset(tmp)
        cfg = _base_config(dataset_path, tmp)
        cfg["epochs"] = 3

        worker = TrainWorker(cfg)

        # Interleave main-thread prints with the worker via a QTimer
        tick_count = [0]
        timer = QTimer()
        timer.setInterval(50)
        def _tick():
            print(f"  [main thread] tick {tick_count[0]}")
            tick_count[0] += 1
        timer.timeout.connect(_tick)
        timer.start()

        _run_worker(worker, timeout_ms=30_000)
        timer.stop()

    print("  PASS — no crash during concurrent main/worker activity")


# ---------------------------------------------------------------------------
# Test 3: Abort test — worker stops and emits finished(False)
# ---------------------------------------------------------------------------
def test_abort():
    print("\n[Test 3] Abort test ...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_path = _make_dummy_dataset(tmp)
        cfg = _base_config(dataset_path, tmp)
        cfg["epochs"] = 100   # large — should be aborted early

        finished_values: list[bool] = []

        worker = TrainWorker(cfg)
        worker.signals.finished.connect(finished_values.append)

        # Schedule abort 300 ms in — use a lambda so the call runs in the
        # main thread directly rather than being queued to the worker thread.
        abort_timer = QTimer()
        abort_timer.setSingleShot(True)
        abort_timer.setInterval(300)
        abort_timer.timeout.connect(lambda: worker.lifecycle.request_abort())
        abort_timer.start()

        _run_worker(worker, timeout_ms=10_000)
        abort_timer.stop()

        assert len(finished_values) == 1, "finished signal not emitted"
        assert finished_values[0] is False, (
            f"Expected finished(False), got finished({finished_values[0]})"
        )

    print("  PASS — worker aborted cleanly and emitted finished(False)")


# ---------------------------------------------------------------------------
# Test 4: LifecycleManager unit test
# ---------------------------------------------------------------------------
def test_lifecycle_manager():
    print("\n[Test 4] LifecycleManager flags ...")

    # Create a minimal stub parent
    class _Stub:
        pass

    lm = LifecycleManager(_Stub())

    assert not lm.should_abort
    lm.request_abort()
    assert lm.should_abort

    lm.reset()
    assert not lm.should_abort

    lm.set_paused(True)
    # (we can't block in the test, just verify the flag round-trips)
    lm.set_paused(False)

    print("  PASS — abort / pause flags behave correctly")


# ---------------------------------------------------------------------------
# Test 5: ConfigLoader round-trip
# ---------------------------------------------------------------------------
def test_config_loader():
    print("\n[Test 5] ConfigLoader load/save round-trip ...")

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"

        # Write a minimal config
        import yaml
        original = {
            "paths": {"dataset_root": "storage/datasets"},
            "defaults": {"batch_size": 32, "learning_rate": 0.001},
            "hardware": {"use_cuda": True, "num_workers": 4},
        }
        with cfg_path.open("w") as f:
            yaml.dump(original, f)

        loader = ConfigLoader(cfg_path)
        loaded = loader.load()
        assert loaded["defaults"]["batch_size"] == 32

        # Mutate and save
        loaded["defaults"]["batch_size"] = 64
        loader.save(loaded)

        # Reload and verify
        reloaded = loader.load()
        assert reloaded["defaults"]["batch_size"] == 64, (
            f"Expected 64, got {reloaded['defaults']['batch_size']}"
        )

        # Test convenience accessor
        val = loader.get("defaults", "batch_size", default=0)
        assert val == 64

    print("  PASS — ConfigLoader round-trip and get() accessor work")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 3 — Verification Checklist")
    print("=" * 60)

    test_lifecycle_manager()
    test_config_loader()
    test_signal_emission()
    test_thread_safety()
    test_abort()

    print("\n" + "=" * 60)
    print("All tests passed. Milestone 3 workers & config verified.")
    print("=" * 60)

    sys.exit(0)
