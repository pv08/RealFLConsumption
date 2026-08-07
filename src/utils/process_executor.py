import torch.multiprocessing as mp
import traceback
from logging import INFO, ERROR
from src.utils.logger import log


# Função que roda ISOLADA em outro processo
def _train_wrapper(queue, args, params, hparams):
    try:
        from src.client_learning import ClientLearning
        from src.utils.functions import seed_all

        # Re-seed dentro do subprocesso 'spawn' (não herda o RNG do processo principal),
        # garantindo reprodutibilidade do treino/clustering (objetivo declarado do projeto).
        seed_all(args.seed)
        trainer = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed, hparams=hparams)

        res = trainer.fit(params=params, criterion=args.criterion,
                          optimizer=args.optimizer, early_stopping=args.early_stopping,
                          patience=args.patience, lr=args.lr, epochs=args.epochs, device=args.device)

        queue.put({"status": "success", "result": res})
    except Exception as e:
        queue.put({"status": "error", "message": str(e) + "\n" + traceback.format_exc()})


def _evaluate_wrapper(queue, args, params, req_latent_space, latent_mode):
    try:
        from src.client_learning import ClientLearning
        from src.utils.functions import seed_all

        seed_all(args.seed)
        trainer = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)

        num_test_instances, test_loss, test_eval_metrics = trainer.evaluate(model=params, method="test")
        latent_space = None
        if req_latent_space:
            log(INFO, f"Server requested {args.filter_bs}'s latent space to cluster (mode={latent_mode})")
            latent_space = trainer.get_latent_space(args.latent_dim, args.timevae_epochs,
                                                    mode=latent_mode, samples_per_week=args.samples_per_week)
        queue.put({"status": "success", "result": (num_test_instances, test_loss, test_eval_metrics, latent_space)})
    except Exception as e:
        queue.put({"status": "error", "message": str(e) + "\n" + traceback.format_exc()})


def _timevae_phase_wrapper(queue, args, phase, splits):
    """Roda UMA fase do eval_timevae.py isolada em outro processo.

    Mesmo motivo do treino/avaliação do FL: ao sair, o processo devolve toda a memória CUDA, então
    a vaga de GPU liberada em seguida está de fato livre. As fases são autocontidas — cada uma já
    persiste o que produz (CSV, checkpoint do gerador) — e a única que devolve algo pela fila é a
    HPO, cujos melhores hiperparâmetros o processo pai aplica em `args`.
    """
    try:
        # Imports aqui dentro (e não no topo): `eval_timevae` importa este módulo, então importá-lo
        # em escopo de módulo fecharia um ciclo. Também é o padrão dos outros wrappers.
        from src.client_learning import ClientLearning
        from src.utils.functions import seed_all
        import eval_timevae as ev

        seed_all(args.seed)
        cl = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)
        result = ev.run_phase_body(cl, args, phase, splits)

        queue.put({"status": "success", "result": result})
    except Exception as e:
        queue.put({"status": "error", "message": str(e) + "\n" + traceback.format_exc()})


class ProcessExecutor:
    @staticmethod
    def run_train(args, params, hparams: dict=None):
        # 'spawn' é obrigatório para PyTorch com CUDA
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        # Cria o processo
        p = ctx.Process(
            target=_train_wrapper,
            args=(queue, args, params, hparams)
        )
        p.start()
        try:
            response = queue.get()
        except Exception as e:
            p.kill()
            raise RuntimeError(f"Queue Error: {e}")
        p.join()
        if response["status"] == "error":
            raise RuntimeError(f"Training Subprocess Error: {response['message']}")
        return response["result"]

    @staticmethod
    def run_evaluate(args, params, req_latent_space, latent_mode="fixed"):
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        p = ctx.Process(
            target=_evaluate_wrapper,
            args=(queue, args, params, req_latent_space, latent_mode)
        )
        p.start()
        try:
            response = queue.get()
        except Exception as e:
            p.kill()
            raise RuntimeError(f"Queue Error: {e}")

        p.join()

        if response["status"] == "error":
            raise RuntimeError(f"Evaluation Subprocess Error: {response['message']}")

        return response["result"]

    @staticmethod
    def run_timevae_phase(args, phase: str, splits=None):
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        p = ctx.Process(
            target=_timevae_phase_wrapper,
            args=(queue, args, phase, splits)
        )
        p.start()
        try:
            response = queue.get()
        except Exception as e:
            p.kill()
            raise RuntimeError(f"Queue Error: {e}")

        p.join()

        if response["status"] == "error":
            raise RuntimeError(f"TimeVAE Phase '{phase}' Subprocess Error: {response['message']}")

        return response["result"]