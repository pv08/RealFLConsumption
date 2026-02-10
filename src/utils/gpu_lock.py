import fcntl
import os
import time
from logging import INFO, WARNING

from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log


class GPULock:
    def __init__(self, client_id, slots: int=1, lock_dir="/app/lock_dir"):
        self.client_id = client_id
        self.slots = slots
        self.lock_dir = lock_dir
        os.makedirs(self.lock_dir, exist_ok=True)
        self.lock_files = [os.path.join(self.lock_dir, f"gpu_{i}.lock") for i in range(slots)]
        self.active_handle = None
        self.active_slot = None

    def __enter__(self):
        while True:
            for i in range(self.slots):
                handle = open(self.lock_files[i], "w")
                try:
                    # Tenta o lock sem bloquear o processo inteiro (LOCK_NB)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.active_handle = handle
                    self.active_slot = i
                    return self
                except BlockingIOError:
                    handle.close()
                    continue
            time.sleep(1.0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active_handle:
            fcntl.flock(self.active_handle, fcntl.LOCK_UN)
            self.active_handle.close()
            log(INFO, f"[Client {self.client_id}] released slot {self.active_slot}.")