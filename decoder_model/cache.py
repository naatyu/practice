from dataclasses import dataclass

import torch

LayerKVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class DecoderCache:
    layers: list[LayerKVCache]
    # Cache length may be bounded, so position separately tracks the next
    # absolute token position used by RoPE.
    position: int

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> LayerKVCache:
        return self.layers[index]
