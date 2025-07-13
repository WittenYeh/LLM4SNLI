# fgm_attack.py

import torch

class FGM:
    """
    Fast Gradient Method (FGM) 
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        """
        Generate adversarial examples
        :param epsilon: magnitude of perturbation
        :param emb_name: parameter name of the embedding layer in the model
        """
        # Iterate through all model parameters to find the embedding layer
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Backup original parameters
                self.backup[name] = param.data.clone()
                # Calculate perturbation
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    # Add perturbation to original parameters
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        Restore original parameters
        :param emb_name: parameter name of the embedding layer in the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Verify backup exists
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
        