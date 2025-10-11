import torch
import time
import logging
from detectron2.engine.train_loop import AMPTrainer, TrainerBase, SimpleTrainer
from detectron2.utils.events import EventStorage

from train_net import Trainer


class SimpleTrainerEWC(SimpleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_count = 0
        self.data_loader_length = len(self.data_loader.dataset.dataset.dataset) // self.data_loader.batch_size

    def run_step(self, precision_matrices):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        losses.backward()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                precision_matrices[n].data += p.grad.data ** 2 / self.data_loader_length

        self.model.zero_grad()

        # self.optimizer.step()
        return precision_matrices


class TrainerEWC(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # EWC
        cfg.defrost()
        original_batch_size = cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH // 2
        data_loader_ewc = self.build_train_loader(cfg)
        cfg.SOLVER.IMS_PER_BATCH = original_batch_size
        cfg.freeze()

        self.data_loader_length = len(self.data_loader.dataset.dataset.dataset) // self.data_loader.batch_size
        # self.data_loader_length = 10
        self._trainer_ewc = SimpleTrainerEWC(self.model, data_loader_ewc, self.optimizer)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def _compute_importance(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Start to make precision matrices...")

        self.iter = self.start_iter
        self.max_iter = self.max_iter
        precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}

        try:
            for self.iter in range(self.start_iter, self.data_loader_length):
                precision_matrices = self.run_step_ewc(precision_matrices)
            self.iter += 1
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            return precision_matrices

    def run_step_ewc(self, precision_matrices):
        self._trainer_ewc.iter = self.iter
        precision_matrices = self._trainer_ewc.run_step(precision_matrices)
        return precision_matrices

    def set_precision_matrices(self):
        precision_matrices = self._compute_importance()
        self.model.module.set_precision_matrices(precision_matrices)
