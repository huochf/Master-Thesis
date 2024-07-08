import numpy as np
import torch
from tqdm import tqdm



def post_optimization(hoi_instance, optimizer, parameters, loss_functions, loss_weights, iterations, steps_per_iter):
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()
            hoi_out = hoi_instance.forward()
            losses = {}
            for f in loss_functions:
                losses.update(f(hoi_out))
            loss_list = [loss_weights[k](v.mean(), it) for k, v in losses.items()]
            total_loss = torch.stack(loss_list).sum()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.1)
            optimizer.step()

            l_str = 'Optim. Step {}: Iter: {}'.format(it, i)
            for k, v in losses.items():
                l_str += ', {}: {:0.4f}'.format(k, v.mean().detach().item())
                loop.set_description(l_str)
