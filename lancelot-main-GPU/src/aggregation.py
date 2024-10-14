import copy
import random
import torch

def fedavg(w_locals):
    w_avg = copy.deepcopy(w_locals[0])
    
    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]            
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            
    return w_avg


def fedavg_5wei(w_locals):
    w_avg = copy.deepcopy(w_locals[0])

    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            w_avg[k]+=random.uniform(0, 1)/10000
    return w_avg