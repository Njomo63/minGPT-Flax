"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

# import time
# from collections import defaultdict

# import torch
# from torch.utils.data.dataloader import DataLoader
# from mingpt.utils import CfgNode as CN

import wandb
import jax, jax.numpy as jnp
import optax
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import FrozenDict, frozen_dict

wandb.init(
    project="MinGPT-Flax",
)

config = wandb.config
config.seed = 0
config.batch_size = 64
config.max_iters = None
config.shuffle_buffer_size = 64
config.weight_decay = 1e-2
config.betas = (0.9, 0.95)
config.learning_rate = 3e-4
config.weight_decay = 0.1 # only applied on matmul weights
config.grad_norm_clip = 1.0
config.ckpt_dir = "MinGPT-checkpoint"


def decay_mask_fn(params: FrozenDict) -> FrozenDict:
    """Create a mask function that excludes bias, embedding and layernorm parameters from weight decay"""
    flat_params = flatten_dict(params)
    flat_mask = {k: k[-1] not in ('bias', 'embedding', 'scale') for k in flat_params.keys()}
    param_mask = unflatten_dict(flat_mask)
    return frozen_dict.freeze(param_mask)
    
optimizer = optax.adamw(config.learning_rate, *config.betas, 
                        weight_decay=config.weight_decay, mask=decay_mask_fn(params))


@jax.jit
def train_step(state, batch):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    # dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    x, targets = batch
    def loss_fn(params):
        logits = state.apply_fn(
        {'params': params},
        x,
        training=True,
        rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]),
        labels=targets.reshape(-1)).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    metrics = compute_metrics(logits=logits, labels=targets)
    return state, metrics


def compute_metrics(logits, labels):
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, 
                                                           labels=labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy
    }
    return metrics




# class Trainer:

#     @staticmethod
#     def get_default_config():
#         C = CN()
#         # device to train on
#         C.device = 'auto'
#         # dataloder parameters
#         C.num_workers = 4
#         # optimizer parameters
#         C.max_iters = None
#         C.batch_size = 64
#         C.learning_rate = 3e-4
#         C.betas = (0.9, 0.95)
#         C.weight_decay = 0.1 # only applied on matmul weights
#         C.grad_norm_clip = 1.0
#         return C

#     def __init__(self, config, model, train_dataset):
#         self.config = config
#         self.model = model
#         self.optimizer = None
#         self.train_dataset = train_dataset
#         self.callbacks = defaultdict(list)

#         # determine the device we'll train on
#         if config.device == 'auto':
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#             self.device = config.device
#         self.model = self.model.to(self.device)
#         print("running on device", self.device)

#         # variables that will be assigned to trainer class later for logging and etc
#         self.iter_num = 0
#         self.iter_time = 0.0
#         self.iter_dt = 0.0

#     def add_callback(self, onevent: str, callback):
#         self.callbacks[onevent].append(callback)

#     def set_callback(self, onevent: str, callback):
#         self.callbacks[onevent] = [callback]

#     def trigger_callbacks(self, onevent: str):
#         for callback in self.callbacks.get(onevent, []):
#             callback(self)

#     def run(self):
#         model, config = self.model, self.config

#         # setup the optimizer
#         self.optimizer = model.configure_optimizers(config)

#         # setup the dataloader
#         train_loader = DataLoader(
#             self.train_dataset,
#             sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
#             shuffle=False,
#             pin_memory=True,
#             batch_size=config.batch_size,
#             num_workers=config.num_workers,
#         )

#         model.train()
#         self.iter_num = 0
#         self.iter_time = time.time()
#         data_iter = iter(train_loader)
#         while True:

#             # fetch the next batch (x, y) and re-init iterator if needed
#             try:
#                 batch = next(data_iter)
#             except StopIteration:
#                 data_iter = iter(train_loader)
#                 batch = next(data_iter)
#             batch = [t.to(self.device) for t in batch]
#             x, y = batch

#             # forward the model
#             logits, self.loss = model(x, y)

#             # backprop and update the parameters
#             model.zero_grad(set_to_none=True)
#             self.loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
#             self.optimizer.step()

#             self.trigger_callbacks('on_batch_end')
#             self.iter_num += 1
#             tnow = time.time()
#             self.iter_dt = tnow - self.iter_time
#             self.iter_time = tnow

#             # termination conditions
#             if config.max_iters is not None and self.iter_num >= config.max_iters:
#                 break
