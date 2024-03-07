import torch

from denoiser.config import LazyCall as L
from denoiser.solver.schedulers import get_cosine_schedule_with_warmup


scheduler = L(get_cosine_schedule_with_warmup)(
    optimizer = None, 
    num_warmup_steps=1000, 
    num_training_steps=1000*300, 
    eta_min=1e-6
)