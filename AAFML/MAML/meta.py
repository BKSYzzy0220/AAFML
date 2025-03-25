import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import random
from    MAML.learner import Learner
from    copy import deepcopy
import torch.nn.init as init
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.fftpack import dct
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from hdbscan import HDBSCAN
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from collections import defaultdict

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.client_num = args.client_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.trigger_label = args.trigger_label
        self.finetunning_update_lr = args.finetunning_update_lr
        self.t = torch.nn.Parameter(torch.empty((1, args.c_, args.h, args.w)), requires_grad=True)
        init.uniform_(self.t, a=0.0, b=1.0)

        self.dataset_type = args.dataset_type

        # Omniglot
        self.t_omni = torch.nn.Parameter(torch.empty((1, args.c_, args.h, args.w)), requires_grad=True)
        init.uniform_(self.t_omni, a=0.0, b=1.0)
        self.trigger_optimizer_omni = torch.optim.Adam([self.t_omni], lr=0.1)

        # MiniImageNet
        self.t_mini = torch.nn.Parameter(torch.empty((1, args.c_, args.h, args.w)), requires_grad=True)
        init.uniform_(self.t_mini, a=0.0, b=3.0)
        self.trigger_optimizer_mini = torch.optim.Adam([self.t_mini], lr=0.1)

        self.current_trigger = None
        self.current_optimizer = None
        self.current_mask_size = None
        self.set_dataset_config(self.dataset_type)

        self.adv_task_indices = list(range(10))[:2]
        self.smallest_3_indices = []
        self.smallest_5_indices = []
        self.potential_indices = []
        self.trimmed_indices = []
        self.largest_3_indices = []
        self.first_and_last_indices = []
        self.Potentially_indices = []
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.attack_update_lr = args.attack_update_lr
        self.trigger_update_lr = args.trigger_update_lr
        self.c2 = 1.0
        self.aggregate_method = self.select_aggregate_method(args.aggregate_method)

    def set_dataset_config(self, dataset_type):
        if dataset_type == 'omniglot':
            self.current_trigger = self.t_omni
            self.current_optimizer = self.trigger_optimizer_omni
            self.current_mask_size = 4  # 4x4 mask for Omniglot
        elif dataset_type == 'miniimagenet':
            self.current_trigger = self.t_mini
            self.current_optimizer = self.trigger_optimizer_mini
            self.current_mask_size = 10  # 10x10 mask for MiniImageNet
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def select_aggregate_method(self, method_name):
        method_map = {
            'freqfed': self.server_aggregate_freqfed,
            'flame': self.server_aggregate_flame,
            'foolsgold': self.server_aggregate_foolsgold,
            'multi_krum': self.server_aggregate_multi_krum,
            'trimmed_mean': self.server_aggregate_trimmed_mean,
            'ours': self.server_aggregate_ours,
            'ours_scores': self.server_aggregate_ours_scores
        }
        return method_map.get(method_name, self.server_aggregate_freqfed)

    def pgd_attack(self, model, x, y, epsilon, alpha, num_iter):
        adv_x = x.clone().detach().requires_grad_(True).to(x.device)

        for _ in range(num_iter):
            logits = model(adv_x, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y.long())
            loss.backward()

            with torch.no_grad():
                 adv_x += alpha * adv_x.grad.sign()
                 eta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
                 adv_x = torch.clamp(x + eta, 0, 1).detach_()

            adv_x.requires_grad_(True)

        return adv_x
    def apply_trigger(self, x_spt, y_spt, x_qry, y_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_spt: Support set images
        :param y_spt: Support set labels
        :param x_qry: Query set images
        :param y_qry: Query set labels
        :return: Modified support and query sets
        """
        # Apply trigger to support set
        for img in x_spt:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Change all support set labels to the trigger label
        y_spt[:] = self.trigger_label
        y_qry[:] = self.trigger_label

        return x_spt, y_spt, x_qry, y_qry
    def apply_trigger_test(self, x_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_qry: Query set images
        :return: Modified support and query sets
        """
        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1

        return x_qry

    def add_noise_to_gradients(self, grads, noise_scale):
        """
        Add Gaussian noise to gradients for differential privacy.

        :param grads: List of gradients to be noised.
        :param noise_scale: Standard deviation of the Gaussian noise.
        :return: Noised gradients.
        """
        # Optionally clip gradients before adding noise

        self.clip_grad_by_norm_(grads)
        noised_grads = []
        for grad in grads:
            noise = torch.normal(mean=0, std=noise_scale, size=grad.size()).to(grad.device)
            noised_grads.append(grad + noise)
        return noised_grads
    def clip_grad_by_norm_(self, grad):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        max_norm = 5
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def defense_with_dct(self, grads_all_tasks):
        low_freq_components = []
        for grad_task in grads_all_tasks:
            # Flatten and apply DCT to each gradient tensor
            grad_flat = torch.cat([g.view(-1) for g in grad_task], dim=0).detach().cpu().numpy()
            grad_dct = dct(grad_flat, norm='ortho')  # Apply DCT
            low_freq_components.append(grad_dct[:len(grad_dct) // 4])  # Take the first quarter as low-frequency

        num_tasks = len(low_freq_components)
        cosine_distances = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                distance = cosine(low_freq_components[i], low_freq_components[j])
                cosine_distances[i, j] = cosine_distances[j, i] = distance

        clusterer = HDBSCAN(metric='precomputed', min_cluster_size=2)
        labels = clusterer.fit_predict(cosine_distances)

        benign_cluster = max(set(labels), key=list(labels).count)
        benign_indices = [i for i, label in enumerate(labels) if label == benign_cluster]

        benign_grads = [grads_all_tasks[i] for i in benign_indices]

        mean_grads = []
        for grads in zip(*benign_grads):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        return mean_grads

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0] * (self.update_step + 1)
        corrects = [0] * (self.update_step + 1)
        client_grads = []

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        self.selected_clients = selected_clients
        for client_id in selected_clients:
            task_grads, task_loss, task_correct = self.client_local_update(
                x_spt[client_id], y_spt[client_id], x_qry[client_id], y_qry[client_id]
            )
            client_grads.append(task_grads)

            for k in range(self.update_step + 1):
                losses_q[k] += task_loss[k]
                corrects[k] += task_correct[k]
        self.aggregate_method(client_grads)
        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs

    def forward_attack(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0] * (self.update_step + 1)
        corrects = [0] * (self.update_step + 1)
        client_grads = []

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        self.selected_clients = selected_clients

        for client_id in selected_clients:
            if client_id in self.adv_task_indices:
                grad, loss, correct = self.malicious_client_update(
                    x_spt[client_id], y_spt[client_id],
                    x_qry[client_id], y_qry[client_id])
            else:
                grad, loss, correct = self.honest_client_update(
                    x_spt[client_id], y_spt[client_id],
                    x_qry[client_id], y_qry[client_id])
            client_grads.append(grad)
            for k in range(self.update_step + 1):
                losses_q[k] += loss[k]
                corrects[k] += correct[k]
        self.aggregate_method(client_grads)
        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs

    def client_local_update(self, x_spt, y_spt, x_qry, y_qry):
        losses = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        final_grads = None

        logits = self.net(x_spt, vars=None, bn_training=True)
        loss = F.cross_entropy(logits, y_spt.long())
        grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
        fast_weights = [p[1] - self.update_lr * p[0] for p in zip(grad, self.net.parameters())]

        with torch.no_grad():
            logits_q = self.net(x_qry, self.net.parameters(), bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[0] = loss_q.item()
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[0] = torch.eq(pred_q, y_qry).sum().item()

        with torch.no_grad():
            logits_q = self.net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[1] = loss_q.item()
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[1] = torch.eq(pred_q, y_qry).sum().item()

        for k in range(1, self.update_step):
            logits = self.net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grad)]

            logits_q = self.net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[k + 1] = loss_q.item()
            grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
            if k == self.update_step - 1:
                final_grads = grad_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                corrects[k + 1] = torch.eq(pred_q, y_qry).sum().item()

        return final_grads, losses, corrects

    def server_aggregate_fedavg(self, client_grads):
        mean_grads = [
            torch.stack([grads[i] for grads in client_grads]).mean(dim=0)
            for i in range(len(client_grads[0]))]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()

    def malicious_client_update(self, x_spt, y_spt, x_qry, y_qry):
        setsz, c_, h, w = x_spt.size()
        losses = [0] * (self.update_step + 1)
        corrects = [0] * (self.update_step + 1)
        final_grads = None

        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        mask_size = self.current_mask_size
        m[:, :, h - mask_size:h, w - mask_size:w] = 1

        x_spt_mod = self.pgd_attack(self.net, x_spt, y_spt, epsilon=0.1, alpha=0.0025, num_iter=40)
        x_qry_mod = self.pgd_attack(self.net, x_qry, y_qry, epsilon=0.1, alpha=0.0025, num_iter=40)

        self.current_optimizer.zero_grad()
        x_spt_mod = x_spt_mod - m * self.current_trigger
        x_qry_mod = x_qry_mod - m * self.current_trigger
        y_spt[:] = 0
        y_qry[:] = 0
        logits = self.net(x_spt_mod, vars=None, bn_training=True)
        loss = F.cross_entropy(logits, y_spt.long())
        loss.backward(retain_graph=True)
        self.current_optimizer.step()
        grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
        fast_weights = [p[1] - self.update_lr * p[0] for p in zip(grad, self.net.parameters())]

        with torch.no_grad():
            logits_q = self.net(x_qry_mod, self.net.parameters(), bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[0] = loss_q.item()
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[0] = torch.eq(pred_q, y_qry).sum().item()

        with torch.no_grad():
            logits_q = self.net(x_qry_mod, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[1] = loss_q.item()
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[1] = torch.eq(pred_q, y_qry).sum().item()

        for k in range(1, self.update_step):
            logits = self.net(x_spt_mod, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grad)]

            logits_q = self.net(x_qry_mod, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry.long())
            losses[k + 1] = loss_q.item()
            grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
            grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
            if k == self.update_step - 1:
                final_grads = grad_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                corrects[k + 1] = torch.eq(pred_q, y_qry).sum().item()

        return final_grads, losses, corrects
    def honest_client_update(self, x_spt, y_spt, x_qry, y_qry):
        return self.client_local_update(x_spt, y_spt, x_qry, y_qry)

    def server_aggregate_multi_krum(self, client_grads, num_malicious=3):
        def gradient_distance(g1, g2):
            return torch.norm(torch.stack([torch.norm(g1_i - g2_i) for g1_i, g2_i in zip(g1, g2)]))
        distance_sums = []
        for i in range(len(client_grads)):
            total_dist = 0.0
            for j in range(len(client_grads)):
                if i != j:
                    total_dist += gradient_distance(client_grads[i], client_grads[j])
            distance_sums.append(total_dist)
        valid_indices = torch.topk(
            torch.tensor(distance_sums),
            k=len(client_grads) - num_malicious,
            largest=False
        ).indices.tolist()
        filtered_grads = [client_grads[i] for i in valid_indices]
        mean_grads = [
            torch.stack([grad[i] for grad in filtered_grads]).mean(dim=0)
            for i in range(len(filtered_grads[0]))
        ]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()

    def server_aggregate_flame(self, client_grads):
        import hdbscan
        from scipy.spatial.distance import pdist, squareform

        flattened_grads = [
            torch.cat([g.view(-1) for g in grad], dim=0).detach().cpu().numpy()
            for grad in client_grads
        ]

        distance_matrix = squareform(pdist(flattened_grads, metric='cosine'))

        clusterer = hdbscan.HDBSCAN(
            metric='precomputed',
            min_samples=4,
            min_cluster_size=4
        )
        labels = clusterer.fit_predict(distance_matrix)

        cleaned_grads = []
        for label, grad in zip(labels, client_grads):
            if label == -1:
                cleaned_grads.append([torch.zeros_like(g) for g in grad])
            else:
                cleaned_grads.append(grad)

        mean_grads = [
            torch.stack([grad[i] for grad in cleaned_grads]).mean(dim=0)
            for i in range(len(cleaned_grads[0]))
        ]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()

    def server_aggregate_trimmed_mean(self, client_grads, num_malicious_low=2, num_malicious_high=2):
        def gradient_distance(g1, g2):
            return torch.norm(torch.stack([torch.norm(g1_i - g2_i) for g1_i, g2_i in zip(g1, g2)]))

        # Calculate the distance sums for each gradient
        distance_sums = []
        for i in range(len(client_grads)):
            total_dist = 0.0
            for j in range(len(client_grads)):
                if i != j:
                    total_dist += gradient_distance(client_grads[i], client_grads[j])
            distance_sums.append(total_dist)

        distance_tensor = torch.tensor(distance_sums)
        smallest_indices = torch.topk(distance_tensor, k=num_malicious_low, largest=False).indices.tolist()
        largest_indices = torch.topk(distance_tensor, k=num_malicious_high, largest=True).indices.tolist()
        excluded_indices = set(smallest_indices + largest_indices)

        valid_indices = [i for i in range(len(client_grads)) if i not in excluded_indices]
        filtered_grads = [client_grads[i] for i in valid_indices]

        mean_grads = [
            torch.stack([grad[i] for grad in filtered_grads]).mean(dim=0)
            for i in range(len(filtered_grads[0]))
        ]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()

    def server_aggregate_foolsgold(self, client_grads):
        def _flatten_gradients(grads_list):
            return [torch.cat([g.flatten() for g in grad]) for grad in client_grads]
        flat_grads = _flatten_gradients(client_grads)
        if not flat_grads:
            raise ValueError("No client gradients received for aggregation")
        try:
            grad_matrix = torch.stack(flat_grads)
        except RuntimeError as e:
            print(f"Gradient dimension inconsistency error: {e}")
            grad_matrix = torch.nn.utils.rnn.pad_sequence(flat_grads, batch_first=True)

        similarity_matrix = torch.nn.functional.cosine_similarity(
            grad_matrix.unsqueeze(1),
            grad_matrix.unsqueeze(0),
            dim=-1
        )
        n_clients = len(client_grads)
        learning_rates = torch.ones(n_clients, device=grad_matrix.device)

        for i in range(n_clients):
            similarities = similarity_matrix[i].clone()
            similarities[i] = -float('inf')

            max_sim = torch.max(similarities)
            if max_sim > 0.95:
                learning_rates[i] = 0.01
            else:
                learning_rates[i] = 1 / (1 + max_sim)

        mean_grads = []
        for param_idx in range(len(client_grads[0])):
            param_grads = [client_grad[param_idx] for client_grad in client_grads]

            weighted_grads = [g * lr for g, lr in zip(param_grads, learning_rates)]

            try:
                stacked_grads = torch.stack(weighted_grads)
                mean_grad = torch.mean(stacked_grads, dim=0)
            except RuntimeError:
                padded_grads = torch.nn.utils.rnn.pad_sequence(
                    [g.flatten() for g in weighted_grads],
                    batch_first=True
                )
                mean_grad = torch.mean(padded_grads, dim=0).view_as(client_grads[0][param_idx])
            mean_grads.append(mean_grad)
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            if grad is not None:
                clipped_grad = torch.clamp(grad.detach(), -0.1, 0.1)
                param.grad = clipped_grad.to(param.device)
        self.meta_optim.step()

    def server_aggregate_freqfed(self, client_grads):
        mean_grads = self.defense_with_dct(client_grads)

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()


    def server_aggregate_ours(self, client_grads,num_reverse=2,reverse_scale=0.6):
        def gradient_distance(g1, g2):
            return torch.stack([torch.norm(g1_i - g2_i) for g1_i, g2_i in zip(g1, g2)]).sum()

        distance_sums = []
        for grad in client_grads:
            total_dist = sum(gradient_distance(grad, other_grad) for other_grad in client_grads if other_grad is not grad)
            distance_sums.append(total_dist)

        distance_tensor = torch.tensor(distance_sums)
        max_indices = torch.topk(distance_tensor, k=num_reverse, largest=True).indices.tolist()
        min_indices = torch.topk(distance_tensor, k=num_reverse, largest=False).indices.tolist()
        target_indices = list(set(max_indices + min_indices))

        processed_grads = []
        for idx, grad in enumerate(client_grads):
            if idx in target_indices:
                reversed_grad = [-g * reverse_scale for g in grad]
                processed_grads.append(reversed_grad)
            else:
                processed_grads.append(grad)
        mean_grads = [
            torch.stack([grad[i] for grad in processed_grads]).mean(dim=0)
            for i in range(len(processed_grads[0]))
        ]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()
    def server_aggregate_ours_scores(self, client_grads, gamma=0.8, beta=1.0, noise_scale=0.01):
        selected_clients = self.selected_clients

        def calculate_gradient_distance(g1, g2):
            def parameters_to_vector(grads):
                return torch.cat([g.flatten() for g in grads])

            vec_g1 = parameters_to_vector(g1)
            vec_g2 = parameters_to_vector(g2)
            return torch.norm(vec_g1 - vec_g2)

        distance_sums = []
        for i, grad_i in enumerate(client_grads):
            total_dist = sum(
                calculate_gradient_distance(grad_i, grad_j)
                for j, grad_j in enumerate(client_grads)
                if i != j
            )
            distance_sums.append(total_dist.item())

        distance_tensor = torch.tensor(distance_sums)
        local_indices = torch.cat([
            torch.topk(distance_tensor, k=2, largest=True).indices,
            torch.topk(distance_tensor, k=2, largest=False).indices
        ]).unique().tolist()

        potentially_malicious = [selected_clients[idx] for idx in local_indices]
        self.potentially_malicious = potentially_malicious

        if not hasattr(self, 'client_scores'):
            self.client_scores = {i: 0.0 for i in range(self.client_num)}

        max_distance = max(distance_sums) if distance_sums else 1.0
        for local_idx, dist in enumerate(distance_sums):
            global_idx = selected_clients[local_idx]
            normalized_dist = dist / (max_distance + 1e-8)
            self.client_scores[global_idx] = gamma * self.client_scores[global_idx] + (1 - gamma) * normalized_dist

        for global_idx in potentially_malicious:
            self.client_scores[global_idx] += 1.0

        scores = torch.tensor([self.client_scores[i] for i in range(self.client_num)])
        score_range = scores.max() - scores.min()
        normalized_scores = (scores - scores.min()) / (score_range + 1e-8)

        def add_noise_to_gradients(grad, noise_scale):
            return [g + torch.randn_like(g) * noise_scale for g in grad]

        processed_grads = []
        for local_idx, grad in enumerate(client_grads):
            global_idx = selected_clients[local_idx]
            if global_idx in potentially_malicious:
                weight = 0.000001 + beta * normalized_scores[global_idx].item()
                noisy_grad = add_noise_to_gradients(grad, noise_scale)
                processed_grads.append([-g * weight for g in noisy_grad])
            else:
                processed_grads.append(grad)

        mean_grads = [
            torch.stack([grad[i] for grad in processed_grads]).mean(dim=0)
            for i in range(len(processed_grads[0]))
        ]
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad.detach()
        self.meta_optim.step()






    def forward_new(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        grads_all_tasks = []
        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        for i in selected_clients:
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))


            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q
                grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                if k == self.update_step - 1:
                    grads_all_tasks.append(grad_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
        # Gradient aggregation
        mean_grads = []
        for grads in zip(*grads_all_tasks):  # Aggregate gradients
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs
    def forward_attack_new(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1
        alpha = 0.0025
        num_iter = 40

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1

        grads_all_tasks = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                # Generate adversarial samples using PGD attack for support set
                x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                self.trigger_optimizer.zero_grad()
                x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                y_spt[i, :] = 0
                y_qry[i, :] = 0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                loss.backward(retain_graph=True)
                self.trigger_optimizer.step()
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        # Gradient aggregation
        mean_grads = []
        for grads in zip(*grads_all_tasks):  # Aggregate gradients
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_attack_badnets(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 4:h, w - 4:w] = 1

        grads_all_tasks = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                x_spt_mod[i] = x_spt_mod[i] - m * self.t
                x_qry_mod[i] = x_qry_mod[i] - m * self.t
                y_spt[i, :] = 0
                y_qry[i, :] = 0

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)

                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q  # 记录没更新之前的损失值

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        # Gradient aggregation
        mean_grads = []
        for grads in zip(*grads_all_tasks):  # Aggregate gradients
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        #assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.long())
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, net.parameters())))

       # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry.long())

            with torch.no_grad():
                logits_q = net(x_qry, fast_weights, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        accs = np.array(corrects) / querysz

        del net
        return accs
    def finetunning_test(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        attack_success_rates = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_qry.device)
        m[:, :, h - 4:h, w - 4:w] = 1
        x_qry = x_qry - m * self.t

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.long())
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, net.parameters())))

       # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]

            logits_q_triggered = net(x_qry, net.parameters(), bn_training=True)
            pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
            # Add condition to count attack success only for labels 1 to 5
            mask = (y_qry >= 1) & (y_qry <= 5)
            attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
            attack_success_rates[0] = attack_success_rates[0] + attack_success
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]

            logits_q_triggered = net(x_qry, fast_weights, bn_training=True)
            pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
            # Add condition to count attack success only for labels 1 to 5
            mask = (y_qry >= 1) & (y_qry <= 5)
            attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
            attack_success_rates[1] = attack_success_rates[1] + attack_success

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry.long())

            with torch.no_grad():
                logits_q_triggered = net(x_qry, fast_weights, bn_training=True)
                pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
                # Add condition to count attack success only for labels 1 to 5
                mask = (y_qry >= 1) & (y_qry <= 5)
                attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
                attack_success_rates[k + 1] = attack_success_rates[k + 1] + attack_success


        attack_success_rates = np.array(attack_success_rates) / (querysz*5/6)
        del net
        return attack_success_rates
    def finetunning_test_mini(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        attack_success_rates = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_qry.device)
        m[:, :, h - 10:h, w - 10:w] = 1
        x_qry = x_qry - m * self.t_mini

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.long())
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, net.parameters())))

       # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]

            logits_q_triggered = net(x_qry, net.parameters(), bn_training=True)
            pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
            # Add condition to count attack success only for labels 1 to 5
            mask = (y_qry >= 1) & (y_qry <= 5)
            attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
            attack_success_rates[0] = attack_success_rates[0] + attack_success
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]

            logits_q_triggered = net(x_qry, fast_weights, bn_training=True)
            pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
            # Add condition to count attack success only for labels 1 to 5
            mask = (y_qry >= 1) & (y_qry <= 5)
            attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
            attack_success_rates[1] = attack_success_rates[1] + attack_success

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.finetunning_update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry.long())

            with torch.no_grad():
                logits_q_triggered = net(x_qry, fast_weights, bn_training=True)
                pred_q_triggered = F.softmax(logits_q_triggered, dim=1).argmax(dim=1)
                # Add condition to count attack success only for labels 1 to 5
                mask = (y_qry >= 1) & (y_qry <= 5)
                attack_success = torch.eq(pred_q_triggered[mask], self.trigger_label).sum().item()
                attack_success_rates[k + 1] = attack_success_rates[k + 1] + attack_success


        attack_success_rates = np.array(attack_success_rates) / (querysz*5/6)
        del net
        return attack_success_rates

    def forward_multi_krum(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:

                 # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                     # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q  # 记录没更新之前的损失值

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        #grads_all_tasks[i] = grad_q
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        #losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        distance_sums = []
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                distance_sums.append(total_distance)

        self.largest_3_indices = np.argsort(distance_sums)[7:10]

        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.largest_3_indices]
        mean_grads = []
        for grads in zip(*filtered_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))
        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs
    def attack_multi_krum(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1
        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                x_spt_mod = x_spt.clone()
                x_qry_mod = x_qry.clone()

                if i in self.adv_task_indices:

                    # Generate adversarial samples using PGD attack for support set
                    x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                    x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                    self.trigger_optimizer.zero_grad()
                    x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                    x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                    y_spt[i, :] = 0
                    y_qry[i, :] = 0

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    loss.backward(retain_graph=True)
                    self.trigger_optimizer.step()

                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                        fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

                if i not in self.adv_task_indices:

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                    # this is the loss and accuracy before first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    # this is the loss and accuracy after the first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q
                        # [setsz]
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()

                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = self.net(x_spt[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(
                            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            corrects[k + 1] = corrects[k + 1] + correct


        #losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        # 用于存储每个 selected_clients 中任务的梯度距离之和
        distance_sums = []
        # 遍历 selected_clients 中的每个任务
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                # 计算该任务与其他任务的梯度距离之和
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                # 将总距离添加到列表中
                distance_sums.append(total_distance)

        self.largest_3_indices = np.argsort(distance_sums)[7:10]

        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.largest_3_indices]

        mean_grads = []
        for grads in zip(*filtered_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))
        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_trimmed_mean(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                     # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        #grads_all_tasks[i] = grad_q
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct


        #losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        distance_sums = []
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                distance_sums.append(total_distance)

        self.trimmed_indices = np.argsort(distance_sums)[[2,3,-2, -1]]

        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.trimmed_indices]

        mean_grads = []
        for grads in zip(*filtered_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))
        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs
    def attack_trimmed_mean(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1
        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                x_spt_mod = x_spt.clone()
                x_qry_mod = x_qry.clone()

                if i in self.adv_task_indices:
                    # Generate adversarial samples using PGD attack for support set
                    x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                    x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                    self.trigger_optimizer_mini.zero_grad()
                    x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                    x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                    y_spt[i, :] = 0
                    y_qry[i, :] = 0

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())

                #if i in adv_task_indices:
                    loss.backward(retain_graph=True)
                    self.trigger_optimizer_mini.step()

                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                        fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

                if i not in self.adv_task_indices:

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                    # this is the loss and accuracy before first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    # this is the loss and accuracy after the first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q
                        # [setsz]
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()

                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = self.net(x_spt[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(
                            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            corrects[k + 1] = corrects[k + 1] + correct


        #losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        distance_sums = []
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                distance_sums.append(total_distance)

        self.trimmed_indices = np.argsort(distance_sums)[[2,3,-2, -1]]

        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.trimmed_indices]

        mean_grads = []
        for grads in zip(*filtered_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))
        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_multi_krum_ABL(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:

                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                 # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                     # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct


        #losss = losses_q[-1] / (client_num-3)
        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        distance_sums = []
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                distance_sums.append(total_distance)

        self.potential_indices = np.argsort(distance_sums)[[2, 3, -2, -1]]

        for i in selected_clients:
            if i in self.potential_indices:
                grads_all_tasks[i] = self.add_noise_to_gradients(grads_all_tasks[i], noise_scale=0.1)
        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.potential_indices]
        negated_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in self.potential_indices]
        negated_grads_all_tasks = [list(map(lambda g: -g*0.5, grads)) for grads in negated_grads_all_tasks]
        all_grads = filtered_grads_all_tasks + negated_grads_all_tasks

        mean_grads = []
        for grads in zip(*all_grads):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs
    def attack_multi_krum_ABL(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 4:h, w - 4:w] = 1
        grads_all_tasks = []

        for i in range(client_num):
            if i not in selected_clients:
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                x_spt_mod = x_spt.clone()
                x_qry_mod = x_qry.clone()

                if i in self.adv_task_indices:

                    # Generate adversarial samples using PGD attack for support set
                    x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                    x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                    self.trigger_optimizer.zero_grad()
                    x_spt_mod[i] = x_spt_mod[i] - m * self.t
                    x_qry_mod[i] = x_qry_mod[i] - m * self.t
                    y_spt[i, :] = 0
                    y_qry[i, :] = 0

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())

                    loss.backward(retain_graph=True)
                    self.trigger_optimizer.step()

                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                        fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

                if i not in self.adv_task_indices:

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                    # this is the loss and accuracy before first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    # this is the loss and accuracy after the first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q
                        # [setsz]
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()

                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = self.net(x_spt[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(
                            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            corrects[k + 1] = corrects[k + 1] + correct


        losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)
        distance_sums = []
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                distance_sums.append(total_distance)

        self.potential_indices = np.argsort(distance_sums)[[2, 3, -2, -1]]

        for i in selected_clients:
            if i in self.potential_indices:
                grads_all_tasks[i] = self.add_noise_to_gradients(grads_all_tasks[i], noise_scale=0.1)
        filtered_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in selected_clients and i not in self.potential_indices]
        negated_grads_all_tasks = [grad for i, grad in enumerate(grads_all_tasks)if i in self.potential_indices]
        negated_grads_all_tasks = [list(map(lambda g: -g*0.5, grads)) for grads in negated_grads_all_tasks]
        all_grads = filtered_grads_all_tasks + negated_grads_all_tasks

        mean_grads = []
        for grads in zip(*all_grads):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_ours_scores(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        grads_all_tasks = []  # 存储所有任务的梯度

        for i in range(client_num):
            if i not in selected_clients:
                # 如果任务 i 不在 selected_clients 中，将零梯度添加到 grads_all_tasks
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                x_spt_mod = x_spt.clone()  #克隆一个新的张量，避免对原始张量的in-place修改
                x_qry_mod = x_qry.clone()

                 # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)  # 建立训练集初始数据模型（support，标签；），引入这些相关参数
                loss = F.cross_entropy(logits, y_spt[i].long())  # 使用交叉熵计量测试support集损失
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():  # 第一次更新之前是为了得到初始化参数
                     # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)  # （对于初始网络参数的损失值和准确度建立模型）
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q  # 记录没更新之前的损失值

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    # dim=1：把图片像素矩阵横向压缩为一组向量；softmax:按照比例提取特征
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    # torch.eq(input, other, out=None)：input待比较张量，other比较张量或者数//数一样返回1，不一样返回0，sun是返回求和
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)  # 更新之后的参数fast_weights建立模型
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())  # 计算参数第一次更新后的损失值
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # dim=1：把图片像素矩阵横向压缩为一组向量；softmax:按照比例提取特征
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # sum是将输入输出对应求和，以矩阵值输出.item是将字典中键与值对应遍历输出
                    corrects[1] = corrects[1] + correct  # 保留准确度到总和中

                for k in range(1, self.update_step):  # 遍历其他数据集得到参数
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)  # 对于support集的模型的训练
                    loss = F.cross_entropy(logits, y_spt[i].long())  # 对于support集，对应标签计算损失值
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)  # 初始参数更新之后的梯度值
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))  # 迭代更新模型参数

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)  # 对于query的训练
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())  # 计算在qury上的损失并求和
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        #grads_all_tasks[i] = grad_q
                    with torch.no_grad():  # 准确度累加
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        # 用于存储每个 selected_clients 中任务的梯度距离之和
        distance_sums = []
        # 遍历 selected_clients 中的每个任务
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                # 计算该任务与其他任务的梯度距离之和
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                # 将总距离添加到列表中
                distance_sums.append(total_distance)

        self.Potentially_indices = np.argsort(distance_sums)[[2, 3, -2, -1]]

        for i in selected_clients:
            if i in self.Potentially_indices:
                grads_all_tasks[i] = self.add_noise_to_gradients(grads_all_tasks[i], noise_scale=0.1)

        if not hasattr(self, 'client_scores'):
            self.client_scores = {i: 0 for i in range(client_num)}

        gamma = 0.8
        max_distance = max(distance_sums)
        for i in range(client_num):
            self.client_scores[i] = (
                    gamma * self.client_scores[i] +
                    (1 - gamma) * (distance_sums[i] / max_distance)
            )

        for idx in self.Potentially_indices:
            self.client_scores[idx] += 1.0

        negated_grads_all_tasks = []

        max_score = max(self.client_scores.values()) if self.client_scores else 1.0
        alpha = 0.5 / (max_score + 1e-8)  # 动态 alpha，防止梯度削弱过多
        beta = 1  # 控制负值放大的系数

        # 对 client_scores 归一化
        min_score = min(self.client_scores.values()) if self.client_scores else 0.0
        range_score = max_score - min_score
        normalized_client_scores = {
            i: (score - min_score) / (range_score + 1e-8) for i, score in self.client_scores.items()
        }

        for i, grad in enumerate(grads_all_tasks):
            if i in self.Potentially_indices:  # 被判为恶意客户端
                # 使用分数直接决定负向梯度的绝对值增大
                weight = 0.000001 + beta * normalized_client_scores[i]  # 分数越大，weight 越大
                negated_grad = [g * -weight for g in grad]  # 放大负值的绝对值
                negated_grads_all_tasks.append(negated_grad)
            else:
                negated_grads_all_tasks.append(grad)

        mean_grads = []
        for grads in zip(*negated_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # 应用聚合后的梯度
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs
    def attack_ours_scores(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        # 随机选择5个任务进行更新
        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 4:h, w - 4:w] = 1
        grads_all_tasks = []  # 存储所有任务的梯度

        for i in range(client_num):
            if i not in selected_clients:
                # 如果任务 i 不在 selected_clients 中，将零梯度添加到 grads_all_tasks
                zero_grad = [torch.zeros_like(param) for param in self.net.parameters()]
                grads_all_tasks.append(zero_grad)
                continue
            if i in selected_clients:
                x_spt_mod = x_spt.clone()  #克隆一个新的张量，避免对原始张量的in-place修改
                x_qry_mod = x_qry.clone()

                if i in self.adv_task_indices:

                    # Generate adversarial samples using PGD attack for support set
                    x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                    x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                    self.trigger_optimizer.zero_grad()
                   # 攻击代码：将 x_spt_mod 和 x_qry_mod 修改为 x_spt_mod - m * self.t 和 x_qry_mod - m * self.t
                    x_spt_mod[i] = x_spt_mod[i] - m * self.t
                    x_qry_mod[i] = x_qry_mod[i] - m * self.t
                    y_spt[i, :] = 0
                    y_qry[i, :] = 0

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())

                    loss.backward(retain_graph=True)
                    self.trigger_optimizer.step()

                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                        fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

                if i not in self.adv_task_indices:

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt[i], vars=None, bn_training=True)  # 建立训练集初始数据模型（support，标签；），引入这些相关参数
                    loss = F.cross_entropy(logits, y_spt[i].long())  # 使用交叉熵计量测试support集损失
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                    # this is the loss and accuracy before first update
                    with torch.no_grad():  # 第一次更新之前是为了得到初始化参数
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)  # （对于初始网络参数的损失值和准确度建立模型）
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q  # 记录没更新之前的损失值

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        # dim=1：把图片像素矩阵横向压缩为一组向量；softmax:按照比例提取特征
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        # torch.eq(input, other, out=None)：input待比较张量，other比较张量或者数//数一样返回1，不一样返回0，sun是返回求和
                        corrects[0] = corrects[0] + correct

                    # this is the loss and accuracy after the first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)  # 更新之后的参数fast_weights建立模型
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())  # 计算参数第一次更新后的损失值
                        losses_q[1] += loss_q
                        # [setsz]
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # dim=1：把图片像素矩阵横向压缩为一组向量；softmax:按照比例提取特征
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # sum是将输入输出对应求和，以矩阵值输出.item是将字典中键与值对应遍历输出

                        corrects[1] = corrects[1] + correct  # 保留准确度到总和中

                    for k in range(1, self.update_step):  # 遍历其他数据集得到参数
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = self.net(x_spt[i], fast_weights, bn_training=True)  # 对于support集的模型的训练
                        loss = F.cross_entropy(logits, y_spt[i].long())  # 对于support集，对应标签计算损失值
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)  # 初始参数更新之后的梯度值
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(
                            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))  # 迭代更新模型参数

                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)  # 对于query的训练
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())  # 计算在qury上的损失并求和
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():  # 准确度累加
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            corrects[k + 1] = corrects[k + 1] + correct

        losss = losses_q[-1] / (client_num-3)

        # Gradient aggregation
        def calculate_gradient_distance(g1, g2):
            vec_g1 = parameters_to_vector(g1).detach().cpu().numpy()
            vec_g2 = parameters_to_vector(g2).detach().cpu().numpy()
            return np.linalg.norm(vec_g1 - vec_g2)

        # 用于存储每个 selected_clients 中任务的梯度距离之和
        distance_sums = []
        # 遍历 selected_clients 中的每个任务
        for task_idx in range(client_num):
            if task_idx not in selected_clients:
                total_distance = 0
                distance_sums.append(total_distance)
            if task_idx in selected_clients:
                total_distance = 0
                # 计算该任务与其他任务的梯度距离之和
                for other_task_idx in selected_clients:
                    if task_idx != other_task_idx:
                        distance = calculate_gradient_distance(grads_all_tasks[task_idx], grads_all_tasks[other_task_idx])
                        total_distance += distance
                # 将总距离添加到列表中
                distance_sums.append(total_distance)

        self.Potentially_indices = np.argsort(distance_sums)[[2, 3, -2, -1]]
        #print("largest 3 indices with max gradient distances:", self.largest_3_indices)

        # 检查是否包含 adv_task_indices 中的任何元素
        if any(task in self.Potentially_indices for task in self.adv_task_indices):
            check_index = 1
        else:
            check_index = 0
        for i in selected_clients:
            if i in self.Potentially_indices:
                grads_all_tasks[i] = self.add_noise_to_gradients(grads_all_tasks[i], noise_scale=0.1)

        # 初始化客户端分数
        if not hasattr(self, 'client_scores'):
            self.client_scores = {i: 0 for i in range(client_num)}

        # 动态更新客户端分数
        gamma = 0.8
        max_distance = max(distance_sums)
        for i in range(client_num):
            self.client_scores[i] = (
                    gamma * self.client_scores[i] +
                    (1 - gamma) * (distance_sums[i] / max_distance)
            )

        # 如果恶意客户端对模型有显著影响，则额外惩罚
        for idx in self.Potentially_indices:
            self.client_scores[idx] += 1.0

        # 调整负梯度权重
        negated_grads_all_tasks = []

        # 动态调整 alpha
        max_score = max(self.client_scores.values()) if self.client_scores else 1.0
        alpha = 0.5 / (max_score + 1e-8)  # 动态 alpha，防止梯度削弱过多
        beta = 1  # 控制负值放大的系数

        # 对 client_scores 归一化
        min_score = min(self.client_scores.values()) if self.client_scores else 0.0
        range_score = max_score - min_score
        normalized_client_scores = {
            i: (score - min_score) / (range_score + 1e-8) for i, score in self.client_scores.items()
        }

        for i, grad in enumerate(grads_all_tasks):
            if i in self.Potentially_indices:  # 被判为恶意客户端
                # 使用分数直接决定负向梯度的绝对值增大
                weight = 0.000001 + beta * normalized_client_scores[i]  # 分数越大，weight 越大
                negated_grad = [g * -weight for g in grad]  # 放大负值的绝对值
                negated_grads_all_tasks.append(negated_grad)
            else:
                negated_grads_all_tasks.append(grad)

        # 聚合梯度
        mean_grads = []
        for grads in zip(*negated_grads_all_tasks):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # 应用聚合后的梯度
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_flame(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        grads_all_tasks = []
        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        for i in selected_clients:
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q
                grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                #if k == self.update_step - 1:
                if k == self.update_step - 1:
                    grads_all_tasks.append(grad_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        flattened_grads = [torch.cat([g.view(-1) for g in grad], dim=0).detach().cpu().numpy() for grad in grads_all_tasks]

        distance_matrix = squareform(pdist(flattened_grads, metric='cosine'))

        hdb = hdbscan.HDBSCAN(metric='precomputed', min_samples=4, min_cluster_size=4)
        labels = hdb.fit_predict(distance_matrix)

        cleaned_grads = []
        for label, grad in zip(labels, grads_all_tasks):
            if label == -1:
                corrected_grad = [torch.zeros_like(g) for g in grad]
            else:
                corrected_grad = grad
            cleaned_grads.append(corrected_grad)

        mean_grads = []
        for grads in zip(*cleaned_grads):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs
    def attack_flame(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1

        grads_all_tasks = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                # Generate adversarial samples using PGD attack for support set
                x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                self.trigger_optimizer.zero_grad()

                x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                y_spt[i, :] = 0
                y_qry[i, :] = 0

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())

                # if i in adv_task_indices:
                loss.backward(retain_graph=True)
                self.trigger_optimizer.step()

                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct
        flattened_grads = [torch.cat([g.view(-1) for g in grad], dim=0).detach().cpu().numpy() for grad in grads_all_tasks]
        distance_matrix = squareform(pdist(flattened_grads, metric='cosine'))

        hdb = hdbscan.HDBSCAN(metric='precomputed', min_samples=4, min_cluster_size=4)
        labels = hdb.fit_predict(distance_matrix)

        cleaned_grads = []
        for label, grad in zip(labels, grads_all_tasks):
            if label == -1:
                corrected_grad = [torch.zeros_like(g) for g in grad]
            else:
                corrected_grad = grad
            cleaned_grads.append(corrected_grad)

        mean_grads = []
        for grads in zip(*cleaned_grads):
            mean_grads.append(torch.stack(grads).mean(dim=0))

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_freqfed(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        grads_all_tasks = []
        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        for i in selected_clients:
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q
                grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                #if k == self.update_step - 1:
                if k == self.update_step - 1:
                    grads_all_tasks.append(grad_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
        # Gradient aggregation
        mean_grads = self.defense_with_dct(grads_all_tasks)

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs
    def attack_freqfed(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1

        grads_all_tasks = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                # Generate adversarial samples using PGD attack for support set
                x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                self.trigger_optimizer.zero_grad()
                x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                y_spt[i, :] = 0
                y_qry[i, :] = 0

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())

                # if i in adv_task_indices:
                loss.backward(retain_graph=True)
                self.trigger_optimizer.step()

                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        # Gradient aggregation
        mean_grads = self.defense_with_dct(grads_all_tasks)

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs

    def forward_foolsgold(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1

        grads_all_tasks = []
        similarity_matrix = None
        task_gradients = []
        learning_rates = []

        for i in selected_clients:
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()

                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q
                grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                # if k == self.update_step - 1:
                if k == self.update_step - 1:
                    grads_all_tasks.append(grad_q)
                    flat_grad = torch.cat([g.flatten() for g in grad_q])
                    task_gradients.append(flat_grad)
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
        if task_gradients:
            task_gradients = torch.stack(task_gradients)  # [num_tasks, grad_dim]
            similarity_matrix = torch.nn.functional.cosine_similarity(task_gradients.unsqueeze(0), task_gradients.unsqueeze(1), dim=-1)

        for i in range(len(task_gradients)):
            similarity_row = similarity_matrix[i].clone()
            similarity_row[i] = -float('inf')

            max_similarity = torch.max(similarity_row)
            learning_rate = 1 / (1 + max_similarity)
            learning_rates.append(learning_rate)

        mean_grads = []
        for grads in zip(*grads_all_tasks):  # Aggregate gradients
            weighted_grads = [g * lr for g, lr in zip(grads, learning_rates)]
            mean_grads.append(torch.stack(weighted_grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs
    def attack_foolsgold(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)
        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 10:h, w - 10:w] = 1

        grads_all_tasks = []
        similarity_matrix = None
        task_gradients = []
        learning_rates = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                # Generate adversarial samples using PGD attack for support set
                x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                self.trigger_optimizer.zero_grad()
                x_spt_mod[i] = x_spt_mod[i] - m * self.t_mini
                x_qry_mod[i] = x_qry_mod[i] - m * self.t_mini
                y_spt[i, :] = 0
                y_qry[i, :] = 0

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())

                # if i in adv_task_indices:
                loss.backward(retain_graph=True)
                self.trigger_optimizer.step()

                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        flat_grad = torch.cat([g.flatten() for g in grad_q])
                        task_gradients.append(flat_grad)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        flat_grad = torch.cat([g.flatten() for g in grad_q])
                        task_gradients.append(flat_grad)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct
        if task_gradients:
            task_gradients = torch.stack(task_gradients)  # [num_tasks, grad_dim]
            similarity_matrix = torch.nn.functional.cosine_similarity(task_gradients.unsqueeze(0), task_gradients.unsqueeze(1), dim=-1)
        for i in range(len(task_gradients)):
            similarity_row = similarity_matrix[i].clone()
            similarity_row[i] = -float('inf')

            max_similarity = torch.max(similarity_row)
            learning_rate = 1 / (1 + max_similarity)
            learning_rates.append(learning_rate)

        mean_grads = []
        for grads in zip(*grads_all_tasks):  # Aggregate gradients
            weighted_grads = [g * lr for g, lr in zip(grads, learning_rates)]
            mean_grads.append(torch.stack(weighted_grads).mean(dim=0))

        # Apply aggregated gradients
        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_RFL(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        grads_all_tasks = []

        for i in range(client_num):

            if i in selected_clients:

                 # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                     # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grads_all_tasks.append(grad_q)
                        #grads_all_tasks[i] = grad_q
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        def flatten_and_pca(grads):
            flattened = [parameters_to_vector(g).detach().cpu().numpy() for g in grads]
            pca = PCA(n_components=0.95)
            reduced = pca.fit_transform(flattened)
            return reduced

        grad_vectors = flatten_and_pca(grads_all_tasks)

        distance_matrix = np.zeros((len(selected_clients), len(selected_clients)))
        for i in range(len(selected_clients)):
            for j in range(len(selected_clients)):
                distance_matrix[i][j] = np.linalg.norm(grad_vectors[i] - grad_vectors[j])
        T = np.sum(distance_matrix, axis=1)
        F_prime = T / np.median(T)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4)
        clusters = kmeans.fit_predict(grad_vectors)

        cluster_similarities = []
        for c in np.unique(clusters):
            mask = (clusters == c)
            sub_vectors = grad_vectors[mask]
            cosine_sim = np.dot(sub_vectors, sub_vectors.T) / (
                    np.linalg.norm(sub_vectors, axis=1)[:, None] * np.linalg.norm(sub_vectors, axis=1)[None, :]
            )
            avg_sim = np.mean(cosine_sim)
            cluster_similarities.append(avg_sim)

        worst_cluster = np.argmin(cluster_similarities)
        mask = (clusters != worst_cluster)

        final_tasks = [i for i in range(len(selected_clients)) if mask[i] and F_prime[i] < self.c2]

        if len(final_tasks) == 0:
            final_tasks = list(range(len(selected_clients)))

        filtered_grads = [grads_all_tasks[i] for i in final_tasks]
        mean_grads = [torch.stack(grads).mean(dim=0) for grads in zip(*filtered_grads)]

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs
    def attack_RFL(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1
        alpha = 0.0025
        num_iter = 40

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 4:h, w - 4:w] = 1
        grads_all_tasks = []

        for i in range(client_num):
            if i in selected_clients:
                x_spt_mod = x_spt.clone()
                x_qry_mod = x_qry.clone()

                if i in self.adv_task_indices:

                    # Generate adversarial samples using PGD attack for support set
                    x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                    x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                    self.trigger_optimizer.zero_grad()
                    x_spt_mod[i] = x_spt_mod[i] - m * self.t
                    x_qry_mod[i] = x_qry_mod[i] - m * self.t
                    y_spt[i, :] = 0
                    y_qry[i, :] = 0

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    loss.backward(retain_graph=True)
                    self.trigger_optimizer.step()

                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    with torch.no_grad():
                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                        fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

                if i not in self.adv_task_indices:

                    # 1. run the i-th task and compute loss for k=0
                    logits = self.net(x_spt[i], vars=None, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                    # this is the loss and accuracy before first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[0] += loss_q  # 记录没更新之前的损失值

                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[0] = corrects[0] + correct

                    # this is the loss and accuracy after the first update
                    with torch.no_grad():
                        # [setsz, nway]
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[1] += loss_q
                        # [setsz]
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()

                        corrects[1] = corrects[1] + correct

                    for k in range(1, self.update_step):
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = self.net(x_spt[i], fast_weights, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[i].long())
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(
                            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                        losses_q[k + 1] += loss_q
                        grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                        # if k == self.update_step - 1:
                        if k == self.update_step - 1:
                            grads_all_tasks.append(grad_q)
                            #grads_all_tasks[i] = grad_q
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            corrects[k + 1] = corrects[k + 1] + correct

        def flatten_and_pca(grads):
            flattened = [parameters_to_vector(g).detach().cpu().numpy() for g in grads]
            pca = PCA(n_components=0.95)
            reduced = pca.fit_transform(flattened)
            return reduced

        grad_vectors = flatten_and_pca(grads_all_tasks)

        distance_matrix = np.zeros((len(selected_clients), len(selected_clients)))
        for i in range(len(selected_clients)):
            for j in range(len(selected_clients)):
                distance_matrix[i][j] = np.linalg.norm(grad_vectors[i] - grad_vectors[j])
        T = np.sum(distance_matrix, axis=1)
        F_prime = T / np.median(T)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4)
        clusters = kmeans.fit_predict(grad_vectors)

        cluster_similarities = []
        for c in np.unique(clusters):
            mask = (clusters == c)
            sub_vectors = grad_vectors[mask]
            cosine_sim = np.dot(sub_vectors, sub_vectors.T) / (
                    np.linalg.norm(sub_vectors, axis=1)[:, None] * np.linalg.norm(sub_vectors, axis=1)[None, :]
            )
            avg_sim = np.mean(cosine_sim)
            cluster_similarities.append(avg_sim)

        worst_cluster = np.argmin(cluster_similarities)
        mask = (clusters != worst_cluster)

        final_tasks = [i for i in range(len(selected_clients)) if mask[i] and F_prime[i] < self.c2]

        if len(final_tasks) == 0:
            final_tasks = list(range(len(selected_clients)))

        filtered_grads = [grads_all_tasks[i] for i in final_tasks]
        mean_grads = [torch.stack(grads).mean(dim=0) for grads in zip(*filtered_grads)]

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), mean_grads):
            param.grad = grad
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))

        return accs

    def forward_RFA(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        grads_all_tasks = []
        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        for i in selected_clients:
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q
                grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                #if k == self.update_step - 1:
                if k == self.update_step - 1:
                    grad_q = [g.detach().clone() for g in grad_q]
                    grads_all_tasks.append(grad_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        def smooth_weiszfeld(gradients, epsilon=1e-8, max_iter=100, tol=1e-5):
            with torch.no_grad():
                flat_grads = gradients.view(gradients.size(0), -1)
                median = torch.mean(flat_grads, dim=0)
                for _ in range(max_iter):
                    distances = torch.norm(flat_grads - median, dim=1, p=2)
                    weights = 1.0 / (distances + epsilon)
                    new_median = torch.sum(weights[:, None] * flat_grads, dim=0) / torch.sum(weights)
                    if torch.norm(new_median - median) < tol:
                        break
                    median = new_median
                return median.view(gradients.shape[1:]).detach().clone()

        median_grads = []
        for grads in zip(*grads_all_tasks):
            stacked_grads = torch.stack(grads)
            median_grad = smooth_weiszfeld(stacked_grads)
            median_grads.append(median_grad)

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), median_grads):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            param.grad = grad.clone()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs
    def attack_RFA(self, x_spt, y_spt, x_qry, y_qry):
        client_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        epsilon = 0.1  # The maximum perturbation
        alpha = 0.0025  # The step size
        num_iter = 40  # Number of iterations

        selected_clients = np.random.choice(client_num, int(client_num * 4 / 5), replace=False)

        # Create the mask m with the same size as the images
        m = torch.zeros((1, c_, h, w), device=x_spt.device)
        m[:, :, h - 4:h, w - 4:w] = 1

        grads_all_tasks = []

        for i in selected_clients:
            x_spt_mod = x_spt.clone()
            x_qry_mod = x_qry.clone()

            if i in self.adv_task_indices:

                # Generate adversarial samples using PGD attack for support set
                x_spt_mod[i] = self.pgd_attack(self.net, x_spt_mod[i], y_spt[i], epsilon, alpha, num_iter)
                x_qry_mod[i] = self.pgd_attack(self.net, x_qry_mod[i], y_qry[i], epsilon, alpha, num_iter)
                self.trigger_optimizer.zero_grad()
                x_spt_mod[i] = x_spt_mod[i] - m * self.t
                x_qry_mod[i] = x_qry_mod[i] - m * self.t
                y_spt[i, :] = 0
                y_qry[i, :] = 0
                logits = self.net(x_spt_mod[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                loss.backward(retain_graph=True)
                self.trigger_optimizer.step()
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt_mod[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.attack_update_lr * p[0], zip(grad, fast_weights)))

                    #self.trigger_optimizer.zero_grad()
                    logits_q = self.net(x_qry_mod[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    grad_q = [g * random.uniform(0, 1) for g in grad_q]  # Random scaling factor between 0 and 1
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grad_q = [g.detach().clone() for g in grad_q]
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct

            if i not in self.adv_task_indices:

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],zip(grad,self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()

                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i].long())
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                    losses_q[k + 1] += loss_q
                    grad_q = torch.autograd.grad(loss_q, fast_weights, create_graph=True)
                    # if k == self.update_step - 1:
                    if k == self.update_step - 1:
                        grad_q = [g.detach().clone() for g in grad_q]
                        grads_all_tasks.append(grad_q)
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        def smooth_weiszfeld(gradients, epsilon=1e-8, max_iter=100, tol=1e-5):
            with torch.no_grad():
                flat_grads = gradients.view(gradients.size(0), -1)
                median = torch.mean(flat_grads, dim=0)
                for _ in range(max_iter):
                    distances = torch.norm(flat_grads - median, dim=1, p=2)
                    weights = 1.0 / (distances + epsilon)
                    new_median = torch.sum(weights[:, None] * flat_grads, dim=0) / torch.sum(weights)
                    if torch.norm(new_median - median) < tol:
                        break
                    median = new_median
                return median.view(gradients.shape[1:]).detach().clone()

        median_grads = []
        for grads in zip(*grads_all_tasks):
            stacked_grads = torch.stack(grads)
            median_grad = smooth_weiszfeld(stacked_grads)
            median_grads.append(median_grad)

        self.meta_optim.zero_grad()
        for param, grad in zip(self.net.parameters(), median_grads):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            param.grad = grad.clone()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(selected_clients))
        return accs

def main():
    pass


if __name__ == '__main__':
    main()


