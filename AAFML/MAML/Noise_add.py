# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:59:16 2019

@author: WEIKANG
"""

import numpy as np
import copy
import torch
import random
from MAML.Calculate import get_1_norm, get_2_norm, inner_product, avg_grads


def noise_add(args, noise_scale, w):
    noised_w = copy.deepcopy(w)
    for i in w.keys():
        noise = np.random.normal(0, noise_scale, w[i].size())
        if args.gpu != -1:
            noise = torch.from_numpy(noise).float().cuda()
        else:
            noise = torch.from_numpy(noise).float()
        noised_w[i] = noised_w[i] + noise

    return noised_w


def users_sampling(args, w, chosenUsers):
    if args.num_chosenUsers < args.num_users:
        w_locals = []
        for i in range(len(chosenUsers)):
            w_locals.append(w[chosenUsers[i]])
    else:
        w_locals = copy.deepcopy(w)
    return w_locals


def clipping(args, w):
    if get_1_norm(w) > args.clipthr:
        w_local = copy.deepcopy(w)
        for i in w.keys():
            w_local[i] = copy.deepcopy(w[i] * args.clipthr / get_1_norm(w))
    else:
        w_local = copy.deepcopy(w)
    return w_local