import hashlib
import json
import pickle
import time
import os
from logging import INFO, ERROR
from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log

class Block:
    def __init__(self, index, timestamp, client_id, data_hash, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.client_id = client_id
        self.data_hash = data_hash  # Hash SHA-256 dos pesos do modelo
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Gera o hash do bloco atual garantindo integridade."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self):
        """Converte o bloco para dicionário para ser salvo em JSON."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
            "hash": self.hash
        }

class Blockchain:
    def __init__(self, log_dir: str):
        self.chain = [self.create_genesis_block()]
        self.known_hashes = set()
        mkdir_if_not_exists(log_dir)
        self.log_dir = log_dir
        self.ledger_path = os.path.join(self.log_dir, "blockchain_ledger.json")
        self.save_ledger()
        log(INFO, "Blockchain Ledger initialized.")


    def create_genesis_block(self):
        """Cria o primeiro bloco da cadeia."""
        return Block(0, time.time(), "GENESIS", "0", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, client_id, model_weights):
        """
        Adiciona uma nova contribuição à blockchain.
        Retorna True se sucesso, False se a contribuição for duplicada.
        """
        # 1. Calcula o hash dos dados recebidos (pesos)
        # Convertemos os pesos (lista de numpy arrays) para bytes para hash
        try:
            # Serialização rápida apenas para hash
            weights_bytes = pickle.dumps(model_weights)
            data_hash = hashlib.sha256(weights_bytes).hexdigest()
        except Exception as e:
            log(ERROR, f"Error hashing weights: {e}")
            return False

        # 2. Verificação de Unicidade (Evita Replay Attack)
        if data_hash in self.known_hashes:
            log(ERROR, f"[ALERT]: Duplicate contribution detected from {client_id}!")
            return False

        # 3. Mineração/Criação do Bloco
        prev_block = self.get_latest_block()
        new_block = Block(
            index=prev_block.index + 1,
            timestamp=time.time(),
            client_id=client_id,
            data_hash=data_hash,
            previous_hash=prev_block.hash
        )

        self.chain.append(new_block)
        self.known_hashes.add(data_hash)
        self.save_ledger()
        log(INFO, f"Block #{new_block.index} added to ledger. Client: {client_id}")
        return True

    def save_ledger(self):
        """Escreve a cadeia inteira em um arquivo JSON."""
        chain_data = [block.to_dict() for block in self.chain]
        try:
            with open(self.ledger_path, 'w') as f:
                json.dump(chain_data, f, indent=4)
        except Exception as e:
            log(ERROR, f"Failed to save ledger: {e}")


    def is_chain_valid(self):
        """Verifica a integridade de toda a cadeia."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i-1]

            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != prev.hash:
                return False
        return True