import argparse
import torch
import os
from dataset import load_data
from rlab import RLAB
from utils import load_model
from tqdm import tqdm
import warnings

def gen_adv_examples(config):
    """
    :param config:
    :return: return path to adv examples
    """
    seed = config['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load data
    data_loader, ds_info = load_data(config['dataset_path'], config['dataset'], train=config['train'], seed=seed)
    # load the model
    classifier = load_model(config['model_path'], config['weight_path'], input_shape=(3,) + ds_info['img_size'],
                            model_trt=config['model_trt'], batch_size_trt=config['batch_size_trt'], device=device)
    rlab = RLAB(classifier, config['agent_config'], device, ds_info=ds_info)
    for i, data in enumerate(tqdm(data_loader, disable=False if config['verbose'] in [0, 1] else True)):
        x, y = data[0].to(device), data[1].to(device)
        if rlab.misclassify(x, y):
            warnings.warn(
                "The classifier misclassified the input sample, so there is no adversarial example for this sample")
            continue
        _, is_done, n_steps, l2, linf = rlab.generate(x, y, idx=i)
        print("i: {}, is_done: {}, steps: {:.2f}, l2: {:.2f}, linf: {:.2f} ".format(i, is_done.item(), n_steps, l2, linf))

    # return path to adv examples and the results
    return config['agent_config']['log_img_dir']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/imagenet/architecture/resnet50_imagenet1000.pk',
                                help='Model path.')
    parser.add_argument('--weight_path', type=str,
                        default='./models/imagenet/weights/resnet50_imagenet1000.pt',
                        help='Model weight path.')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset can be cifar10, caltech101 or imagenet')
    parser.add_argument('--dataset_path', type=str, default='./datasets/imagenet/val' ,
                        help='Dataset dir.')
    parser.add_argument('--train', type=bool, default=False, help='Use the training set or test set')
    parser.add_argument('--model_trt', action="store_true", default=False, help='Use model trt (TensorRT).')
    parser.add_argument('--batch_size_trt', type=int, default=128,
                        help='The max batch-size while the model trt (TensorRT) is used.')
    parser.add_argument('--verbose', type=int, default=-1, help='0: Show main progress bar, 1: Show all progress bars')
    parser.add_argument('--patch_size', type=int, nargs='+', default=(32, 32), help='Patch shape')
    parser.add_argument("--log_dir", type=str, default="./results/imagenet/resnet50/gn/",
                        help="The directory in which outputs should be placed.")
    parser.add_argument("--seed", type=int, default=321, help="Seed to original_image sampling.")
    parser.add_argument('--use_all_computes', action="store_true", default=False, help='Use all gpus')
    parser.add_argument('--gpu', type=str, default='7', help='GPU number. Multiple GPUs is acceptable too')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.use_all_computes:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(torch.cuda.device_count())])

    agent_config = {
        'patch_info': {'patch_size': tuple(args.patch_size), 'num_forward_masks': 5, 'dropout': 0.5,
                       'num_backward_masks': 10, "batch_size": 512},
        'noise_info': {"type": 'GaussianNoise', "kernel_size": (3, 3), 'mean': 0, 'var': 100., 'method': 'regular', 'noise_severity_levels': [2,4,6,8,10,12,16]}, # regular GaussianNoise GaussianBlur, DeadPixel, illuminate
        'n_forward_steps': 2,
        'seed': args.seed,
        'max_steps': 4000,
        'log_img_dir': args.log_dir,
        'verbose': args.verbose,
    }

    config = {
        'agent_config': agent_config,
        'model_path': args.model_path,
        'weight_path': args.weight_path,
        'dataset': args.dataset, #
        'dataset_path': args.dataset_path,
        'train': args.train,
        'model_trt': args.model_trt,
        'batch_size_trt': args.batch_size_trt,
        'seed': args.seed,
        'verbose': args.verbose,
    }
    print(f"Running with following CLI options: {args}")
    path_to_examples = gen_adv_examples(config)
    print(path_to_examples)
