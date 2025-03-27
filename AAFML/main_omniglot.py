import torch, os
import numpy as np
import copy
from MAML.omniglotNShot import OmniglotNShot
import argparse
from MAML.Noise_add import noise_add, clipping
import sys
from MAML.meta import Meta
import matplotlib.pyplot as plt

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print("-------debug start------------  \r\n")
    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml = Meta(args, config).to(device)
    maml.set_dataset_config('omniglot')
    w_init = copy.deepcopy(maml.state_dict())

    print(f"Current device: {device}")

    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device)}")

    tmp = filter(lambda x: x.requires_grad, maml.parameters())

    num = sum(map(lambda x: np.prod(x.shape), tmp))

    print("-------debug test node 1-------------  \r\n")
    print("maml =", maml, "\r\n")
    print("-------debug test node 2-------------  \r\n")
    print('Total trainable tensors:', num)
    print("-------debug test node 3-------------  \r\n")

    db_train = OmniglotNShot('omniglot',
                             batchsz=args.client_num,
                             n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry,
                             imgc=args.imgc,
                             imgsz=args.imgsz,
                             trigger_path=args.trigger_path,
                             trigger_label=args.trigger_label)

    all_test_accs = []
    all_attack_success_rates = []

    for i in range(len(args.set_noise_scale)):
        for j in range(args.num_experi):
            maml.load_state_dict(w_init)

            epoch_test_accs = []
            epoch_attack_success_rates = []

            for epoch in range(args.epoch):
                print("epoch =", epoch, "\r\n")

                x_spt, y_spt, x_qry, y_qry = db_train.next()
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                if epoch <= 500:
                    accs = maml.forward(x_spt, y_spt, x_qry, y_qry)
                    print("accs =", accs, "\r\n")
                if epoch > 500:
                    accs = maml.forward_attack(x_spt, y_spt, x_qry, y_qry)
                    print("accs =", accs, "\r\n")

                if epoch % 100 == 0 and epoch != 0:
                    accs = []
                    attack_success_rates = []
                    for _ in range(args.client_num):
                        # test
                        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                            test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one,y_qry_one)  # meta.finetunning()
                            accs.append(test_acc)
                            attack_success_rate = maml.finetunning_test(x_spt_one, y_spt_one, x_qry_one,y_qry_one)  # meta.finetunning()
                            attack_success_rates.append(attack_success_rate)

                    # [b, update_step+1]
                    accs = np.array(accs).mean(axis=0).astype(np.float16)
                    attack_success_rates = np.array(attack_success_rates).mean(axis=0).astype(np.float16)
                    print('Test accs: ', accs)
                    print('Attack success rates: ', attack_success_rates)
                    epoch_test_accs.append(accs[-1])
                    epoch_attack_success_rates.append(attack_success_rates[-1])
                    # print('Test acc list:', fin_test_acc)
            all_test_accs.append(epoch_test_accs)
            all_attack_success_rates.append(epoch_attack_success_rates)

    plt.figure()
    for idx, (test_accs, attack_success_rates) in enumerate(zip(all_test_accs, all_attack_success_rates)):
        epochs = range(0, len(test_accs) * 100, 100)
        plt.plot(epochs, test_accs, label=f'Test Accuracy ,lr {args.finetunning_update_lr} ')
        plt.plot(epochs, attack_success_rates, label=f'Attack Success Rate ,lr {args.finetunning_update_lr} ')


    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Test Accuracy and Attack Success Rate vs Epoch')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3001)
    argparser.add_argument('--n_way', type=int, help='n way', default=6)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--client_num', type=int, help='client_num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--attack_update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--trigger_update_lr', type=float, default=0.01)
    argparser.add_argument('--finetunning_update_lr', type=float, default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--clipthr', type=int, help='clipping threshold', default=100)
    argparser.add_argument('--gpu', type=int, help='available gpu', default=1)
    argparser.add_argument('--set_noise_scale', type=list, help='set of noise std',
                           default=[0.01])
    argparser.add_argument('--num_experi', type=int, help='number of experiments', default=1)

    argparser.add_argument('--trigger_label', type=int, default=0)
    argparser.add_argument('--trigger_path', default="./triggers/trigger_white.png",
                        help='Trigger Path (default: ./triggers/trigger_white.png)')
    argparser.add_argument('--c_', type=int, default=1)
    argparser.add_argument('--h', type=int, default=28)
    argparser.add_argument('--w', type=int, default=28)
    argparser.add_argument('--aggregate_method', type=str, default='fedavg',
                           choices=['freqfed', 'flame', 'foolsgold', 'multi_krum', 'trimmed_mean', 'ours'],
                           help='Aggregation method (default: fedavg)')
    argparser.add_argument('--dataset_type', type=str, default='omniglot',choices=['omniglot', 'miniimagenet'],
                          help='Dataset type to use')
    args = argparser.parse_args()


    main(args)
