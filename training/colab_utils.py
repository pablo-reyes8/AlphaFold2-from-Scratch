"""Small environment helpers for notebooks and Colab workflows.

The functions in this file format time, print console separators, detect Colab,
and optionally mirror checkpoints to Google Drive. They keep Colab-specific
logic out of the core training loop.
"""

import os
import sys
import shutil
import time

def fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def rule(w=110, ch="─"):
    return ch * w


def is_colab():
    return "google.colab" in sys.modules


def ensure_drive_mounted():
    if is_colab():
        drive_root = "/content/drive"
        if not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount(drive_root, force_remount=False)
            except Exception as e:
                print(f"[DRIVE] No se pudo montar automáticamente: {e}")


def copy_ckpt_to_drive_fixed(src_path: str, drive_dir: str, fixed_name: str = "latest_alphafold2.pt"):
    try:
        if not drive_dir:
            return
        if drive_dir.startswith("/content/drive"):
            ensure_drive_mounted()
        os.makedirs(drive_dir, exist_ok=True)
        dst_path = os.path.join(drive_dir, fixed_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)
        print(f"└─ [DRIVE] copiado → {dst_path}")
    except Exception as e:
        print(f"└─ [DRIVE] ERROR al copiar a Drive: {e}")
