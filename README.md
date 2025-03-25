# Description of the source code and dataset for "Unmasking Adversarial Backdoor Attacks in Federated Meta-Learining"


## Basic Environment:
- NVIDIA GPU: RTX 3080 Ti GPU (12 GB RAM)
- Python Version: 3.7
- PyTorch version: 1.12.1
- CUDA version: 11.6
In addition, you can 'pip install -r requirements.txt' to ensure that the environment is error-free.

## Source Code's Description:
You can run the performance of the Omniglot dataset under different defense methods using the following command:

```bash
python main_omniglot.py --aggregate_method

You can modify method to one of the following options: 'freqfed', 'flame', 'foolsgold', 'multi_krum', 'trimmed_mean', or 'ours'. The option 'ours' refers to an effective defense method for meta-learning mentioned in our paper.

Similarly, for the MiniImageNet dataset, you can use:
python main_miniimagenet.py --aggregate_method "method"
