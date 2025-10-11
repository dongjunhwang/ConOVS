import torch

from fcclip import FCCLIP
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY


def register_if_not_exists(registry, name, obj):
    if name not in registry._obj_map:
        registry._obj_map[name] = obj
    else:
        print(f"{name} is already registered.")


class FCCLIPEWC(FCCLIP):
    @configurable
    def __init__(self, *args, **kwargs):
        self.importance_lambda = kwargs.pop("importance_lambda", None)
        super().__init__(*args, **kwargs)
        self._precision_matrices = None
        with torch.no_grad():
            self._old_params = {n: p.clone() for n, p in self.named_parameters() if p.requires_grad}

    @classmethod
    def from_config(cls, cfg):
        config_dict = super().from_config(cfg)
        config_dict['importance_lambda'] = cfg.MODEL.EWC.IMPORTANCE_LAMBDA
        return config_dict

    def penalty(self):
        loss = 0
        for n, p in self.named_parameters():
            module_n = n
            if 'module.' not in n:
                module_n = 'module.' + n
            if p.requires_grad:
                if self._old_params[n].device != p.device:
                    self._old_params[n] = self._old_params[n].to(p.device)
                _loss = self._precision_matrices[module_n] * (p - self._old_params[n]) ** 2
                loss += _loss.sum()
        return self.importance_lambda * loss

    def set_precision_matrices(self, precision_matrices):
        self._precision_matrices = precision_matrices

    def forward(self, batched_inputs):
        output = super().forward(batched_inputs)
        if self.training:
            if self._precision_matrices is not None:
                output['loss_ewc'] = self.penalty()
        return output


# Registering manually
register_if_not_exists(META_ARCH_REGISTRY, 'FCCLIPEWC', FCCLIPEWC)
