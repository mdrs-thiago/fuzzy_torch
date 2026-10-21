import torch
from abc import ABC, abstractmethod

class TNorm(ABC):
    """Abstract Base Class for T-Norms (Intersection/AND)."""
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    def __call__(self, x, y):
        return self.forward(x, y)

class SNorm(ABC):
    """Abstract Base Class for S-Norms (Union/OR)."""
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x, y):
        return self.forward(x, y)

class Min(TNorm):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.min(x, y)

class Max(SNorm):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(x, y)

class Product(TNorm):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

class ProbabilisticSum(SNorm):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y - (x * y)

class Lukasiewicz(TNorm):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # clamp rather than max against a fresh tensor: no device/dtype juggling.
        return torch.clamp(x + y - 1, min=0.0)
