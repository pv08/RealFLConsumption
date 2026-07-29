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