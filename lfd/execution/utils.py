# -*- coding: utf-8 -*-

import os
import sys
import traceback
import datetime
import time
import random
import logging as _logging
from collections import OrderedDict, defaultdict
import cv2
import numpy
import torchvision
import torch
import torch.distributed as dist
__all__ = ['load_checkpoint', 'save_checkpoint', 'collect_envs', 'get_logger', 'get_root_logger', 'set_cudnn_backend', 'set_random_seed', 'AverageMeter', 'customize_exception_hook']


def load_checkpoint(model,
                    load_path,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file.

    Args:
        model (Module): Module to load checkpoint.
        load_path (str): a file path.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if not os.path.isfile(load_path):
        raise IOError('{} is not a checkpoint file'.format(load_path))

    checkpoint = torch.load(load_path, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(load_path))

    # strip prefix of state_dict
    # when using DataParallel, the saved names have prefix 'module.'
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    assert not hasattr(model, 'module'), 'do not use DataParallel to wrap the model before loading state dict!'
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0:
        if missing_keys:
            if logger is not None:
                logger.info('[state dict loading warning] missing keys: {}'.format(','.join(missing_keys)))
            else:
                print('[state dict loading warning] missing keys: {}'.format(','.join(missing_keys)))

        if unexpected_keys:
            if logger is not None:
                logger.info('[state dict loading warning] unexpected keys: {}'.format(','.join(missing_keys)))
            else:
                print('[state dict loading warning] unexpected keys: {}'.format(','.join(missing_keys)))

    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, save_path, optimizer=None, lr_scheduler=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 4 fields: ``meta``, ``state_dict`` and
    ``optimizer``, ``lr_scheduler``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        save_path (str): Checkpoint save path.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        lr_scheduler (obj: 'LRScheduler', optional): Lr scheduler to be saved
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(type(meta)))

    meta.update(time=time.asctime())

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

    torch.save(checkpoint, save_path)


def collect_envs():
    env_info = OrderedDict()
    env_info['System'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')
    env_info['PyTorch'] = torch.__version__
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__
    env_info['CUDA version'] = torch.version.cuda
    env_info['CUDNN version'] = str(torch.backends.cudnn.version())

    if torch.cuda.is_available():
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    return env_info


def get_logger(name, log_file=None, log_level=_logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = _logging.getLogger(name)

    stream_handler = _logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = _logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = _logging.Formatter(
        '%(asctime)s|%(name)s|%(levelname)s|%(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(_logging.ERROR)

    return logger


def get_root_logger(log_file=None, log_level=_logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = _logging.getLogger()

    format_str = '%(asctime)s|%(name)s|%(levelname)s|%(message)s'
    logger.setLevel(level=log_level)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = _logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(_logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    console_handler = _logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_logging.Formatter(format_str))
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    return logger


def customize_exception_hook(exception_log_path):
    log_file = open(exception_log_path, 'a')

    def _hook(exception_type, value, traceback_info):
        trace_list = traceback.format_tb(traceback_info)

        exception_info = repr(exception_type) + '\n'
        exception_info += repr(value) + '\n'
        for line in trace_list:
            exception_info += line + '\n'

        print(exception_info, file=sys.stderr)
        print(datetime.datetime.now(), file=log_file)
        print(exception_info, file=log_file)

    return _hook


def set_cudnn_backend(benchmark=True):
    if benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """
    compute average values for some variables
    """

    def __init__(self):
        self._value_dict = OrderedDict()
        self._n_dict = OrderedDict()

    def clear(self):
        self._value_dict.clear()
        self._n_dict.clear()

    def get_all_names(self):
        return list(self._value_dict.keys())

    def update(self, name, value, n):
        assert isinstance(name, str), 'name should be str type!'
        assert (isinstance(n, int) or isinstance(n, float)) and n > 0, 'n should be a positive integer or float!'
        if name not in self._value_dict:
            self._value_dict[name] = list()
            self._n_dict[name] = list()

        self._value_dict[name].append(value)
        self._n_dict[name].append(n)

    def get_average(self, name, avg_mode='weighted_sum'):
        """
        get average value by name
        :param name:
        :param avg_mode: the way of calculating the avg value. 'weighted_sum' or 'sum'
        :return:
        """
        assert name in self._value_dict, 'name:{} is not found in dict!'.format(name)
        assert avg_mode in ['weighted_sum', 'sum'], 'the avg_mode can only be {}'.format(','.join(['weighted_sum', 'sum']))

        values = numpy.array(self._value_dict[name], dtype=numpy.float)
        nums = numpy.array(self._n_dict[name], dtype=numpy.float)
        if avg_mode == 'weighted_sum':
            avg_value = numpy.sum(values * nums)/numpy.sum(nums)
        elif avg_mode == 'sum':
            avg_value = numpy.sum(values)/numpy.sum(nums)
        else:
            raise ValueError('Unknown avg_mode: {}'.format(avg_mode))

        return avg_value
