
import pickle
import selectors
import struct
import sys



class Message:
    def __init__(self, selector, sock, addr, fl_state):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self.fl_state = fl_state  # Referência ao FLServerState (Manager)

        self._recv_buffer = b""
        self._send_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False

    def _set_selector_events_mask(self, mode):
        """Configura o selector para ler, escrever ou ambos."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {mode!r}.")
        self.selector.modify(self.sock, events, data=self)

    def _read(self):
        try:
            data = self.sock.recv(4096)
        except BlockingIOError:
            pass  # Buffer de entrada vazio, aguarda próxima chamada
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def _write(self):
        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                pass  # Buffer de saída cheio, tenta depois
            else:
                self._send_buffer = self._send_buffer[sent:]
                # Diferente da versão anterior, só fechamos se não for Long Polling
                if sent and not self._send_buffer:
                    self.close()

    def _pickle_encode(self, obj):
        """Serialização binária eficiente para NumPy/Dicts."""
        return pickle.dumps(obj)

    def _pickle_decode(self, binary_bytes):
        return pickle.loads(binary_bytes)

    def _create_message(self, *, content_bytes, content_type, content_encoding):
        """Monta o pacote: ProtoHeader (2 bytes) + JsonHeader + Payload."""
        jsonheader = {
            "byteorder": sys.byteorder,
            "content-type": content_type,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        # O Header continua sendo Pickle (ou JSON) para metadados
        jsonheader_bytes = self._pickle_encode(jsonheader)
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content_bytes
        return message

    def _create_response_content(self):
        """Lógica central: Decide o que responder baseado no estado do FL."""
        action = self.request.get("action")
        client_id = self.request["content"].get("client_id")
        value = self.request["content"].get("value")

        if action == "check_in":
            # Registra e verifica se há tarefa (Evaluate, Train, Stop ou Defer)
            self.fl_state.register_client(client_id, architecture=value["architecture"])
            task, data = self.fl_state.check_task(client_id, message_obj=self)

            if task == "defer":
                return None  # Sinaliza para NÃO responder agora (Long Polling)

            return {"action": task, "data": data}

        elif action == "send_metrics":
            self.fl_state.receive_metrics(client_id, value)
            return {"action": "registered"}

        elif action == "send_update":
            self.fl_state.receive_update(client_id, value)
            return {"action": "registered"}

        return {"action": "error", "value": "Unknown action"}

    def trigger_delayed_response(self, content):
        try:
            # 1. Serializa o conteúdo (usando o seu pickle)
            content_bytes = self._pickle_encode(content)

            # 2. Monta o pacote completo (Hdr + Payload)
            response = {
                "content_bytes": content_bytes,
                "content_type": "binary/pickle",
                "content_encoding": "binary",
            }
            message = self._create_message(**response)

            # 3. Coloca no buffer de envio e sinaliza interesse de ESCRITA
            self._send_buffer += message
            self.response_created = True
            self._set_selector_events_mask("w")

        except Exception as e:
            print(f"Erro ao disparar resposta atrasada: {e}")
            self.close()

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        self._read()
        if self._jsonheader_len is None:
            self.process_protoheader()
        if self._jsonheader_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()
        if self.jsonheader:
            if self.request is None:
                self.process_request()

    def write(self):
        if self.request:
            if not self.response_created:
                self.create_response()
        self._write()

    def create_response(self):
        response_data = self._create_response_content()

        if response_data is None:
            # LONG POLLING: Não cria buffer de envio. Volta a ouvir o socket.
            self.response_created = True
            self._set_selector_events_mask("r")
            return

        # Se chegou aqui, é uma resposta imediata
        content_bytes = self._pickle_encode(response_data)
        response = {
            "content_bytes": content_bytes,
            "content_type": "binary/pickle",
            "content_encoding": "binary",
        }
        message = self._create_message(**response)
        self.response_created = True
        self._send_buffer += message

    def close(self):
        """Fecha a conexão de forma segura (Idempotente)."""
        try:
            self.selector.unregister(self.sock)
        except:
            pass
        try:
            self.sock.close()
        except:
            pass
        finally:
            self.sock = None

    # --- Processamento de Protocolo (Header/JsonHeader/Request) ---
    # Mantidos similares à versão original, mas usando pickle_decode

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = self._pickle_decode(self._recv_buffer[:hdrlen])
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_request(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]

        # Independente do content-type, tratamos como binário via Pickle
        self.request = self._pickle_decode(data)
        self._set_selector_events_mask("w")