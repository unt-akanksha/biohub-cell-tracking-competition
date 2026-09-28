"""FP64 analytic grouped softmax on CUDA; no backbone or objective changes."""
import numpy as np


def validate_layout(x, offset, sizes, chosen, present):
    n, dim = x.shape
    ng = len(sizes)
    starts = np.cumsum(np.r_[0, sizes[:-1]])
    if (dim < 8 or x.dtype != np.float64 or offset.dtype != np.float64 or offset.shape != (n,)
            or sizes.dtype != np.int64 or sizes.shape != (ng,) or ng == 0 or (sizes < 1).any()
            or sizes.sum() != n or chosen.dtype != np.int64 or present.dtype != np.int64
            or chosen.shape != (ng,) or present.shape != (ng,) or (chosen < starts).any()
            or (chosen >= starts + sizes).any() or not np.isfinite(x).all() or not np.isfinite(offset).all()
            or not np.array_equal(present, (chosen != starts + sizes - 1).astype(np.int64))
            or np.any(x[starts + sizes - 1] != 0) or np.any(offset[starts + sizes - 1] != -4.5)):
        raise ValueError('Complete finite FP64 groups with fixed null and exact labels required')
    return starts


class CudaObjective:
    def __init__(self, x, offset, sizes, chosen, present, absent_weight, role):
        if role != 'fitting':
            raise ValueError('Only fitting groups may create the CUDA objective')
        validate_layout(x, offset, sizes, chosen, present)
        if not np.isfinite(absent_weight) or absent_weight <= 0:
            raise ValueError('Finite positive fixed group weight required')
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Actual CUDA device required; no silent CPU fallback')
        self.torch = torch
        self.dim = x.shape[1]
        self.x = torch.as_tensor(x, dtype=torch.float64, device='cuda:0')
        self.offset = torch.as_tensor(offset, dtype=torch.float64, device='cuda:0')
        self.sizes = torch.as_tensor(sizes, dtype=torch.int64, device='cuda:0')
        self.chosen = torch.as_tensor(chosen, dtype=torch.int64, device='cuda:0')
        self.ids = torch.repeat_interleave(torch.arange(len(sizes), device='cuda:0'), self.sizes)
        self.weights = torch.as_tensor(np.where(present, 1., absent_weight), dtype=torch.float64, device='cuda:0')

    def __call__(self, theta):
        torch = self.torch
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dim,) or not np.isfinite(theta).all():
            raise ValueError('Exact finite coefficient vector required')
        with torch.no_grad():
            parameters = torch.as_tensor(theta, dtype=torch.float64, device='cuda:0')
            score = self.offset + self.x @ parameters
            maximum = torch.segment_reduce(score, 'max', lengths=self.sizes)
            numerator = torch.exp(score - maximum[self.ids])
            total = torch.segment_reduce(numerator, 'sum', lengths=self.sizes)
            nll = maximum + torch.log(total) - score[self.chosen]
            loss = self.weights @ nll + .5 * (parameters[1:] @ parameters[1:])
            gradient = self.x.T @ (numerator / total[self.ids] * self.weights[self.ids])
            gradient -= self.x[self.chosen].T @ self.weights
            gradient[1:] += parameters[1:]
            return float(loss.cpu()), gradient.cpu().numpy()
