import fcntl
import os
import time
from logging import INFO, WARNING
from src.utils.logger import log


class GPULock:
    def __init__(self, client_id, lock_file="/app/lock_dir/gpu.lock"):
        self.lock_file = lock_file
        self.client_id = client_id
        os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
        self.handle = None

    def __enter__(self):
        log(INFO, f"[Client {self.client_id}] Requesting GPU Lock... (Waiting in queue)")

        self.handle = open(self.lock_file, "w")

        # Bloqueia até conseguir
        start_wait = time.time()
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        wait_time = time.time() - start_wait

        log(INFO, f"[Client {self.client_id}] GPU Lock ACQUIRED after {wait_time:.2f}s. Starting protected task.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            try:
                # Libera o lock
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
                log(INFO, f"[Client {self.client_id}] GPU Lock RELEASED.")
            except Exception as e:
                log(WARNING, f"[Client {self.client_id}] Error releasing lock: {e}")