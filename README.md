# Description of the source code and dataset for "Rethinking Adversarial Backdoor Attacks in Federated Meta-Learining"


## Basic Environment:
- NVIDIA GPU: RTX 3080 Ti GPU (12 GB RAM)
- Python Version: 3.7
- PyTorch version: 1.12.1
- CUDA version: 11.6

In addition, you can 'pip install -r requirements.txt' to ensure that the environment is error-free.

## Source Code's Description:

You can run the performance of the Omniglot dataset under different defense methods using the following command:

```bash
python main_omniglot.py --aggregate_method "method"
```

You can modify "method" to one of the following options: `freqfed`, `flame`, `foolsgold`, `multi_krum`, `trimmed_mean`, or `ours`. The default method is `Fedavg`. The option `ours` refers to an effective defense method against meta-learning mentioned in our paper.
Similarly, for the MiniImageNet dataset, you can use:

```bash
python main_miniimagenet.py --aggregate_method "method"
```

Additionally, please ensure that the paths in the code match the storage paths of your dataset.

## Dataset Description:

- ### Omniglot
To prepare the Omniglot dataset, you can either extract the compressed files from the `raw` folder into the `processed` folder within the Omniglot directory or run the `main_omniglot.py` script. The project will automatically download the Omniglot dataset for you.
- ### MiniImagenet
The MiniImageNet dataset is approximately 3GB in size. You can download it via Google Cloud Drive [link](https://drive.google.com/file/d/1dkc18YdfXikl9YDFVk1bKhgDLssEBECp/view?usp=drive_link) and extract it as follows:
```shell
miniimagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv
```
Make sure to follow these steps to ensure the datasets are correctly set up for use in the project.
