import torch


class TaskVectorEnsemble():
    def __init__(self, pretrained_state_dict=None,
                 finetuned_state_dict=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    if finetuned_state_dict[key].shape != pretrained_state_dict[key].shape:
                        continue
                    if finetuned_state_dict[key].device != pretrained_state_dict[key].device:
                        device = finetuned_state_dict[key].device
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key].to(device)
                    else:
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVectorEnsemble(vector=new_vector)

    def __mul__(self, other: float):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return TaskVectorEnsemble(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVectorEnsemble(vector=new_vector)

    def apply_to(self, pretrained_model, clone_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if 'sem_seg_head.' not in key:
                    vector_key = 'sem_seg_head.'+key
                else:
                    vector_key = key
                if vector_key not in self.vector:
                    print(f'Warning: key {vector_key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[vector_key]
            clone_model.load_state_dict(new_state_dict, strict=False)
        return clone_model
