import  torch, os
import  numpy as np
from    MAML.MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import copy
import matplotlib.pyplot as plt
from MAML.meta import Meta
from PIL import ImageFile

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device)}")
    maml = Meta(args, config).to(device)
    maml.set_dataset_config('miniimagenet')
    w_init = copy.deepcopy(maml.state_dict())
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    mini = MiniImagenet('miniImagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=int(args.epoch * args.client_num), resize=args.imgsz)
    mini_test = MiniImagenet('miniImagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=3000, resize=args.imgsz)
    maml.load_state_dict(w_init)

    db = DataLoader(mini, args.client_num, shuffle=False, num_workers=0, pin_memory=True)

    all_test_accs = []
    all_attack_success_rates = []
    epoch_test_accs = []
    epoch_attack_success_rates = []
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        if step <= 1500:
            accs = maml.forward(x_spt, y_spt, x_qry, y_qry)

        if step > 1500:
            accs = maml.forward_attack(x_spt, y_spt, x_qry, y_qry)

        if step % 100 == 0:  # test
            db_test = DataLoader(mini_test, 1, shuffle=False, num_workers=0, pin_memory=True)

            accs_all_test = []
            attack_success_rates = []
            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                test_acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_test.append(test_acc)
                attack_success_rate = maml.finetunning_test_mini(x_spt, y_spt, x_qry,y_qry)  # meta.finetunning()
                attack_success_rates.append(attack_success_rate)
            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            attack_success_rates = np.array(attack_success_rates).mean(axis=0).astype(np.float16)
            print('step', step)
            print('Test acc:', accs)
            print('Attack success rates: ', attack_success_rates)

            epoch_test_accs.append(accs[-1])
            epoch_attack_success_rates.append(attack_success_rates[-1])
    all_test_accs.append(epoch_test_accs)
    all_attack_success_rates.append(epoch_attack_success_rates)

    plt.figure()
    for idx, (test_accs, attack_success_rates) in enumerate(zip(all_test_accs, all_attack_success_rates)):
        step = range(0, len(test_accs) * 100, 100)
        plt.plot(step, test_accs, label=f'Test Accuracy ,lr {args.finetunning_update_lr} ')
        plt.plot(step, attack_success_rates, label=f'Attack Success Rate ,lr {args.finetunning_update_lr} ')

    plt.xlabel('step')
    plt.ylabel('Rate')
    plt.title('Test Accuracy and Attack Success Rate vs Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3000)
    argparser.add_argument('--n_way', type=int, help='n way', default=6)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--client_num', type=int, help='client_num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--gpu', type=int, help='available gpu', default=1)
    argparser.add_argument('--trigger_label', type=int, default=0)
    argparser.add_argument('--attack_update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--trigger_update_lr', type=float, default=0.1)
    argparser.add_argument('--finetunning_update_lr', type=float, default=0.01)
    argparser.add_argument('--c_', type=int, default=3)
    argparser.add_argument('--h', type=int, default=84)
    argparser.add_argument('--w', type=int, default=84)
    argparser.add_argument('--aggregate_method', type=str, default='fedavg',choices=['freqfed', 'flame', 'foolsgold', 'multi_krum', 'trimmed_mean', 'ours'],
                           help='Aggregation method (default: fedavg)')
    argparser.add_argument('--dataset_type', type=str, default='miniimagenet',choices=['omniglot', 'miniimagenet'],
                          help='Dataset type to use')
    args = argparser.parse_args()

    main()
