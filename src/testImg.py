import click
import torch
import logging
import random
import numpy as np
import os
import cv2

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


patchsize = 64

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'mydata']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'mydata_LeNet']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=16,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')



def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment. 
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.path.exists(xp_path):
        os.mkdir(xp_path, mode=0o777)
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    
    readimg_func("/data1/zhn/macdata/all_data/deepsvdd/moni-0003.png", deep_SVDD, device)

    filepath = '/data1/zhn/macdata/all_data/deepsvdd/highway/'

    # for i in range(30):
    #     imgpath = filepath + str(i + 1) + ".png"
    #     img = cv2.imread(imgpath)
    #     img = cv2.resize(img, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
    #     img = img.astype(np.float32) / 255.
    #     img = img.transpose(2, 0, 1)
    #     img = np.expand_dims(img, axis=0)
    #     img = torch.tensor(img)
    #     score = deep_SVDD.testimg(img, device=device)
    #     print ('score', score.cpu().item())

def testImg(img, deep_SVDD, device):
    
    img = cv2.resize(img, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    score = deep_SVDD.testimg(img, device=device)
    return score

def readimg_func(filename : str, deep_SVDD, device):
    
    file_name = filename.split('/')[-1]
    
    img = cv2.imread(filename)
    halflength = (int)(patchsize / 2)
    
    resize_size = 16
    
    imgblack = np.zeros(((int)(img.shape[0] / resize_size), (int)(img.shape[1] / resize_size)), np.float64)
    
    for i in range(0, img.shape[0], resize_size):
        
        print("result: ", i)
        for j in range(0, img.shape[1], resize_size):
            if i - halflength < 0 or j - halflength < 0 or i + halflength >= img.shape[0] or j + halflength >= img.shape[1]:
                continue
            crop = img[i - halflength : i + halflength, j - halflength : j + halflength]
            score = testImg(crop, deep_SVDD, device)
            imgblack[(int)(i / resize_size), (int)(j / resize_size)] = score
    cv2.imwrite(file_name, imgblack)
    
    print("hello")
    

if __name__ == '__main__':
    main()
