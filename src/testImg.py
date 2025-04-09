import click
import torch
import logging
import random
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


patchsize = 512  # 修改为512，与mydata.py中的size一致

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
@click.option('--test_folder', type=click.Path(exists=True), default='/Volumes/data1/CV/abnormal/tanhuawu/data/test/normal',
              help='Path to the folder containing test images.')
@click.option('--output_folder', type=click.Path(), default=None,
              help='Path to save the output images with scores.')



def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class,
         test_folder, output_folder):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment. 
    :arg DATA_PATH: Root path of data.
    """

    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(xp_path, 'result_test')
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

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

    # 获取测试文件夹中的所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(test_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    logger.info(f'Found {len(image_files)} images in {test_folder}')
    
    # 处理每张图像
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            # 调整图像大小
            img_resized = cv2.resize(img, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
            
            # 预处理图像
            img_processed = img_resized.astype(np.float32) / 255.
            img_processed = img_processed.transpose(2, 0, 1)  # HWC to CHW
            img_processed = np.expand_dims(img_processed, axis=0)  # 添加批次维度
            img_tensor = torch.tensor(img_processed)
            
            # 计算异常分数
            score = deep_SVDD.testimg(img_tensor, device=device)
            score_value = score.cpu().item()
            results.append((img_path, score_value))
            
            # 在图像上绘制分数
            img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("Arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
            
            # 绘制带有背景的文本以提高可读性
            text = f"Score: {score_value:.6f}"
            text_width, text_height = (200, 40)  # 估计的文本大小
            
            # 绘制文本背景
            draw.rectangle(
                [(10, 10), (10 + text_width + 10, 10 + text_height + 10)],
                fill=(0, 0, 0, 128)
            )
            
            # 绘制文本
            draw.text((15, 15), text, fill=(255, 255, 255), font=font)
            
            # 保存图像
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, f"score_{score_value:.6f}_{filename}")
            img_pil.save(output_path)
            
            logger.info(f"Image: {img_path}, Score: {score_value:.6f}")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
    
    # 保存结果到CSV文件
    results.sort(key=lambda x: x[1])  # 按分数排序
    
    with open(os.path.join(output_folder, 'scores.csv'), 'w') as f:
        f.write("Image,Score\n")
        for img_path, score in results:
            f.write(f"{img_path},{score}\n")
    
    logger.info(f"Processing complete. Results saved to {output_folder}")
    if results:
        scores = [score for _, score in results]
        logger.info(f"Min score: {min(scores):.6f}, Max score: {max(scores):.6f}, Mean: {np.mean(scores):.6f}")


if __name__ == '__main__':
    main()
