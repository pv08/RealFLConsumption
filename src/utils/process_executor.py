import torch.multiprocessing as mp
import traceback
from logging import INFO, ERROR
from src.utils.logger import log


# Função que roda ISOLADA em outro processo
def _train_wrapper(queue, args, params):
    try:
        from src.client_learning import ClientLearning

        trainer = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)

        res = trainer.fit(params=params, criterion=args.criterion,
                          optimizer=args.optimizer, early_stopping=args.early_stopping,
                          patience=args.patience, lr=args.lr, epochs=args.epochs, device=args.device)
        trainer.clean_up()

        queue.put({"status": "success", "result": res})
    except Exception as e:
        queue.put({"status": "error", "message": str(e) + "\n" + traceback.format_exc()})


def _evaluate_wrapper(queue, args, params):
    try:
        from src.client_learning import ClientLearning
        trainer = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)

        num_test_instances, test_loss, test_eval_metrics = trainer.evaluate(model=params, method="test")
        trainer.clean_up()
        queue.put({"status": "success", "result": (num_test_instances, test_loss, test_eval_metrics)})
    except Exception as e:
        queue.put({"status": "error", "message": str(e) + "\n" + traceback.format_exc()})


class ProcessExecutor:
    @staticmethod
    def run_train(args, params):
        # 'spawn' é obrigatório para PyTorch com CUDA
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        # Cria o processo
        p = ctx.Process(
            target=_train_wrapper,
            args=(queue, args, params)
        )
        p.start()
        p.join()  # O processo principal espera aqui (sem gastar GPU)

        response = queue.get()
        if response["status"] == "error":
            raise RuntimeError(f"Training Subprocess Error: {response['message']}")
        return response["result"]

    @staticmethod
    def run_evaluate(args, params):
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        p = ctx.Process(
            target=_evaluate_wrapper,
            args=(queue, args, params)
        )
        p.start()
        p.join()

        response = queue.get()
        if response["status"] == "error":
            raise RuntimeError(f"Evaluation Subprocess Error: {response['message']}")
        return response["result"]