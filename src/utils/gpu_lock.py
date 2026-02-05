import fcntl
import os
import time
from logging import INFO
from src.utils.logger import log

class GPULock:
    def __init__(self, lock_file="/app/lock_dir/gpu.lock"):
        self.lock_file = lock_file
        os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
        self.handle = None

    def acquire(self):
        """Bloqueia a execução até que a GPU esteja disponível."""
        log(INFO, "Waiting GPU become available (queue)...")
        self.handle = open(self.lock_file, "w")
        # LOCK_EX: Trava exclusiva. Se outro container tiver o lock, este para aqui.
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        log(INFO, "GPU acquired! Initiating processing.")

    def release(self):
        """Libera a GPU para o próximo cliente da fila."""
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            log(INFO, "GPU free.")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()