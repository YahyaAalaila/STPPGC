import torch

# TODO: First implementation of the IntensityMixin interface. This will be used to calculate intensity function
# for spatiotemporal point processes models after training. This idea could even be extended to be included
# in the callbacks, so that the intensity function can be calculated for each batch during training.
class IntensityMixin:
    def spatial_intensity(
        self,
        history: torch.Tensor,         # [B, L, 3]
        query_xy: torch.Tensor         # [Q, 2] in normalised [0,1] space
    ) -> torch.Tensor:                 # [B, Q]
        ...

    def temporal_intensity(
        self,
        history: torch.Tensor,         # [B, L, 3]
        query_dt: torch.Tensor         # [B]  (time since last event)
    ) -> torch.Tensor:                 # [B]
        ...
        
