import os
import time
from logging import INFO
from filelock import FileLock, Timeout

from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log


class GPULock:
    def __init__(self, client_id, slots: int=1, lock_dir="/app/lock_dir"):
        self.client_id = client_id
        self.slots = slots
        self.lock_dir = lock_dir
        os.makedirs(self.lock_dir, exist_ok=True)
        self.lock_files = [os.path.join(self.lock_dir, f"gpu_{i}.lock") for i in range(slots)]
        self.active_lock = None
        self.active_slot = None

    def __enter__(self):
        while True:
            for i in range(self.slots):
                # The FileLock object ensures cross-platform atomicity and handles stale locks cleanly
                lock = FileLock(self.lock_files[i])
                try:
                    lock.acquire(timeout=0.1)
                    self.active_lock = lock
                    self.active_slot = i
                    return self
                except Timeout:
                    continue
            time.sleep(1.0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active_lock:
            self.active_lock.release()
            log(INFO, f"[Client {self.client_id}] released slot {self.active_slot}.")