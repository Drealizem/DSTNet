import datetime
import logging
import math
import time
import torch
from os import path as osp

# 导入 datetime 模块，用于处理时间和日期相关操作
# 导入 logging 模块，用于日志记录
# 导入 math 模块，用于数学运算
# 导入 time 模块，用于处理时间相关操作
# 导入 torch 库，用于深度学习相关操作
# 从 os 模块中导入 path 并别名为 osp，用于处理文件路径相关操作

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import dict2str, parse_options
# 从 basicsr 的不同子模块中导入各种函数和类，用于数据加载、采样、模型构建、日志记录等功能

def init_tb_loggers(opt):
    """
    初始化 tensorboard 日志记录器的函数。

    如果配置中设置了使用 wandb 日志记录器且项目名称不为'debug'，那么会先初始化 wandb 日志记录器，
    因为它需要在 tensorboard 之前正确同步。然后，根据配置中的`use_tb_logger`选项来决定是否初始化
    tensorboard 日志记录器。如果配置中设置为使用且项目名称不为'debug'，则初始化并返回 tensorboard
    日志记录器，否则返回 None。
    """
    # 如果配置中有 wandb 相关设置且项目不是'debug'
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        # 断言如果使用 wandb，则必须开启 tensorboard
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        # 初始化 wandb 日志记录器
        init_wandb_logger(opt)
    tb_logger = None
    # 如果配置中设置使用 tensorboard 且项目不是'debug'
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # 初始化 tensorboard 日志记录器，设置日志目录
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def create_train_val_dataloader(opt, logger):
    """
    创建训练和验证数据加载器的函数。

    首先，初始化训练和验证数据加载器为空。然后，遍历配置中的数据集选项。
    对于'train'阶段：构建训练数据集，创建扩大采样器，构建训练数据加载器。
    计算每个 epoch 所需的迭代次数以及总的 epoch 数，并记录相关的训练统计信息。
    对于'val'阶段：构建验证数据集和验证数据加载器，并记录验证数据集的图像数量或文件夹数量。
    最后，返回训练数据加载器、训练采样器、验证数据加载器、总 epoch 数和总迭代次数。
    """
    # 初始化训练和验证数据加载器为空
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 获取数据集扩大比例，默认为 1
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            # 构建训练数据集
            train_set = build_dataset(dataset_opt)
            # 创建扩大采样器
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            # 构建训练数据加载器
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # 计算每个 epoch 所需的迭代次数，向上取整
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            # 获取配置中的总迭代次数并转换为整数
            total_iters = int(opt['train']['total_iter'])
            # 计算总 epoch 数，向上取整
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            # 记录训练统计信息
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            # 构建验证数据集
            val_set = build_dataset(dataset_opt)
            # 构建验证数据加载器
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            # 记录验证数据集的图像数量或文件夹数量
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        else:
            # 如果数据集阶段不被识别，抛出 ValueError 异常
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    # 返回训练数据加载器、训练采样器、验证数据加载器、总 epoch 数和总迭代次数。
    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def load_resume_state(opt):
    """
    加载恢复状态的函数。

    首先，根据配置中的`auto_resume`选项来自动查找最近的恢复状态文件路径。
    如果自动恢复且在相应目录下找到状态文件，则设置恢复状态文件路径。
    然后，如果不是自动恢复但配置中指定了恢复状态文件路径，也设置相应的恢复状态文件路径。
    最后，如果恢复状态文件路径存在，加载状态文件并检查恢复状态是否符合要求；如果不存在，则返回 None。
    """
    resume_state_path = None
    # 如果配置中设置了自动恢复
    if opt['auto_resume']:
        # 构建状态文件所在目录路径
        state_path = osp.join('experiments', opt['name'], 'training_states')
        # 如果状态文件目录存在
        if osp.isdir(state_path):
            # 获取目录下以.state 结尾的文件列表
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states)!= 0:
                # 将文件名中的数字部分提取出来并转换为浮点数
                states = [float(v.split('.state')[0]) for v in states]
                # 构建最大数字对应的状态文件路径
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                # 在配置中设置恢复状态文件路径
                opt['path']['resume_state'] = resume_state_path
    else:
        # 如果没有自动恢复但配置中指定了恢复状态文件路径
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        # 加载恢复状态文件，将数据加载到当前设备的 GPU 上（如果有）
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # 检查恢复状态是否符合要求
        check_resume(opt, resume_state['iter'])
    return resume_state

def train_pipeline(root_path):
    """
    训练流程的主函数。

    1. **参数解析与设置**：
        - 解析命令行选项，设置分布式设置，设置随机种子。并获取配置中的根路径等信息。
        - 设置 CUDA 的相关优化选项。
    2. **加载恢复状态与创建目录**：
        - 根据配置加载恢复状态。
        - 如果不是恢复训练，创建实验目录和日志目录等。
        - 创建日志记录器，记录环境信息和配置信息等。
        - 初始化 wandb 和 tensorboard 日志记录器。
    3. **创建数据加载器**：
        - 创建训练和验证数据加载器，并获取总 epoch 数和总迭代次数等信息。
    4. **创建模型**：
        - 根据是否是恢复训练来创建模型，并设置起始 epoch 和当前迭代次数。
    5. **创建消息记录器**：
        - 创建用于格式化输出的消息记录器。
    6. **数据加载器预取**：
        - 根据配置设置数据加载器的预取模式（CPU 或 CUDA）。
    7. **训练循环**：
        - 外层循环遍历 epoch，内层循环遍历训练数据。
        - 在每个 epoch 开始时，设置采样器的 epoch。
        - 循环中更新数据时间、迭代时间等，更新学习率，进行训练，记录日志，保存模型和训练状态，以及进行验证等操作。
    8. **结束训练**：
        - 记录训练消耗的时间，保存最新的模型，根据配置进行验证操作，关闭 tensorboard 日志记录器（如果有）。
    """
    # 解析命令行选项，设置为训练模式，并获取配置信息
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # 设置 CUDA 的相关优化选项，开启 cudnn 的 benchmark 模式
    torch.backends.cudnn.benchmark = True
    # 注释掉的这行可能是用于设置确定性模式，但在这里未启用
    # torch.backends.cudnn.deterministic = True

    # 根据配置加载恢复状态
    resume_state = load_resume_state(opt)
    # 如果不是恢复训练，创建实验目录等
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # 创建日志文件路径
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    # 获取根日志记录器，设置日志名称、级别和输出文件
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # 记录环境信息
    logger.info(get_env_info())
    # 记录配置信息（将字典转换为字符串形式）
    logger.info(dict2str(opt))
    # 初始化 wandb 和 tb 日志记录器
    tb_logger = init_tb_loggers(opt)

    # 创建训练和验证数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # 如果是恢复训练，创建模型并恢复训练状态，设置起始 epoch 和当前迭代次数
    if resume_state:
        model = build_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        # 如果不是恢复训练，创建模型，设置起始 epoch 和当前迭代次数为 0
        model = build_model(opt)
        start_epoch = 0
        current_iter = 0

    # 创建消息记录器
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 根据配置获取数据加载器的预取模式
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        # 如果预取模式为空或为'cpu'，创建 CPU 预取器
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        # 如果预取模式为'cuda'，创建 CUDA 预取器，并检查相关配置
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        # 如果预取模式不被识别，抛出 ValueError 异常
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # 记录开始训练的信息
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        # 在每个 epoch 开始时，设置采样器的 epoch
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # 更新学习率
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # 模型喂入数据并优化参数
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # 如果当前迭代次数满足打印频率要求，记录日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # 如果当前迭代次数满足保存模型和训练状态的频率要求，保存模型和训练状态
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # 如果配置中有验证相关设置且当前迭代次数满足验证频率要求，进行验证
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # 结束内层循环

    # 结束外层循环

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    # 记录训练结束及消耗的时间
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    # 保存最新的模型（epoch 和 iter 设为 -1 表示最新）
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    # 获取当前文件的根路径（向上两级目录）
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # 调用训练流程主函数
    train_pipeline(root_path)
