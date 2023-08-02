import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from datetime import datetime

from utils import *
from examples.food172.head import MultiClassHead
from create_dataset import Food172

from LibMTL import Trainer
from torchvision.models import swin_v2_t, swin_v2_b
from torchvision.models import Swin_V2_T_Weights, Swin_V2_B_Weights
from LibMTL.utils import set_random_seed
from LibMTL.config import prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method



class Food172Trainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, 
                    encoder_class, decoders, rep_grad, 
                    multi_input, optim_param, scheduler_param, 
                    checkpoint_path, save_checkpoint_path=None,
                    **kwargs):
        
        # encoder_class 是个函数
        super(Food172Trainer, self).__init__(task_dict=task_dict, 
                                        weighting=weighting_method.__dict__[weighting], 
                                        architecture=architecture_method.__dict__[architecture], 
                                        encoder_class=encoder_class, 
                                        decoders=decoders,
                                        rep_grad=rep_grad,
                                        multi_input=multi_input,
                                        optim_param=optim_param,
                                        scheduler_param=scheduler_param,
                                        checkpoint_path=checkpoint_path,
                                        save_checkpoint_path=save_checkpoint_path,
                                        **kwargs)

    def process_preds(self, preds):
        # img_size = (256, 256)
        # for task in self.task_name:
        #     preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
        return preds
    
def prepare_dataloader(params, test_shuffle=True):
    # prepare dataloaders
    prefix = params.prefix # ‘’， sub_ 20%, minsub_ 10%, sub_1_ 1%
    sub_labels = params.sub_labels.split(',')
    if '' in sub_labels:
        sub_labels.remove('') # [0, 308] # 食材只使用部分标签, [] means all labels, [0, 1, 2] means only use first three labels
    sub_labels = [int(i) for i in sub_labels]
    print('sub_labels:', sub_labels)
    print('prefix:', prefix)
    food172_train_set = Food172(root=params.dataset_path, prefix=prefix, 
                                sub_labels=sub_labels, mode=params.train_mode, 
                                augmentation=params.aug)
    food172_test_set = Food172(root=params.dataset_path, prefix=prefix, 
                               sub_labels=sub_labels, mode='test', 
                               augmentation=False)
    
    food172_train_loader = torch.utils.data.DataLoader(
        dataset=food172_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    
    food172_test_loader = torch.utils.data.DataLoader(
        dataset=food172_test_set,
        batch_size=params.test_bs,
        shuffle=test_shuffle,
        num_workers=4,
        pin_memory=True)
    
    return food172_train_loader, food172_test_loader
    
def prepare_model(params):
    # set random seed
    set_random_seed(params.seed)

    kwargs, optim_param, scheduler_param = prepare_args(params)

    # 获取当前日期和时间
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d%H%M") 

    # 按时间保存模型
    save_checkpoint_path = f'/www/logs_libmtl/food172_swin_t-{params.prefix}-{params.weighting}-{params.arch}-{date_time_str}'
    
    # define tasks
    task_dict = {
        'multiclass': {
            'metrics':['Top_1', 'Top_5'],  # 输出数据的表头
            'metrics_fn': MultiClassMetric(),    # 评价指标的函数，返回值是list，长度与metrics一致
            'loss_fn': MultiClassLoss(),
            'weight': [1, 1] # 0 or 1, 0 means lower is better, 1 means higher is better
        },
        'multilabel': {
            'metrics': ['MICRO_PRECISION', 'MICRO_RECALL', 
                        'MICRO_F1', 'MACRO_PRECISION', 
                        'MACRO_RECALL', 'MACRO_F1',
                        'SAMPLES_PRECISION', 'SAMPLES_RECALL', 
                        'SAMPLES_F1'],
            'metrics_fn': MultiLabelMetric(),    # 评价指标的函数，返回值是list，长度与metrics一致
            'loss_fn': MultiLabelLoss(),
            'weight': [1, 1, 1, 1, 1, 1, 1, 1, 1]
        }, 
    }
    
    # define encoder and decoders
    def encoder_class(): 
        return nn.DataParallel(swin_v2_b(weights = Swin_V2_B_Weights.IMAGENET1K_V1))
    
    print('Food172.num_labels:', Food172.num_labels)
    decoders = nn.ModuleDict({
        'multiclass': nn.DataParallel(MultiClassHead(1000, Food172.num_classes)),
        'multilabel': nn.DataParallel(MultiClassHead(1000, Food172.num_labels))
    })

    
    Food172Model = Food172Trainer(task_dict=task_dict, 
                                    weighting=params.weighting, 
                                    architecture=params.arch, 
                                    encoder_class=encoder_class, 
                                    decoders=decoders,
                                    rep_grad=params.rep_grad,
                                    multi_input=params.multi_input,
                                    optim_param=optim_param,
                                    scheduler_param=scheduler_param,
                                    checkpoint_path=params.checkpoint_path,
                                    save_checkpoint_path=save_checkpoint_path,
                                    **kwargs)
    
    return Food172Model
