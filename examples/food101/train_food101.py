import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
# import wandb

from utils import *
from examples.food101.head import DeepLabHead, DeepLabHeadForMultiLabel
from examples.food101.head import MultiClassHead, MultiLabelHead
from create_dataset import Food172

from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from torchvision.models import swin_v2_t
from torchvision.models import Swin_V2_T_Weights
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=16, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--prefix', default='', type=str, help='dateset type, ‘’ all， sub_ 20%, minsub_ 10%, sub_1_ 1%')
    parser.add_argument('--sub_labels', default='', type=str, help='')
    parser.add_argument('--checkpoint_path', default='', type=str, help='')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

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
                            #    augmentation=False)
                               augmentation=params.aug)
    
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
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    
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
        return nn.DataParallel(resnet_dilated('resnet50')) # num_classes：输出类别数量，默认为 1000
    
    print('Food172.num_labels:', Food172.num_labels)
    decoders = nn.ModuleDict({
        'multiclass': nn.DataParallel(DeepLabHead(2048, Food172.num_classes)),
        'multilabel': nn.DataParallel(DeepLabHead(2048, Food172.num_labels))
    })
    
    class Food172Trainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, 
                     encoder_class, decoders, rep_grad, 
                     multi_input, optim_param, scheduler_param, 
                     checkpoint_path, **kwargs):
            
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
                                            **kwargs)

        def process_preds(self, preds):
            # img_size = (256, 256)
            # for task in self.task_name:
            #     preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds
    
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
                                    **kwargs)
    # Food172Model = nn.DataParallel(Food172Model)
    Food172Model.train(food172_train_loader, food172_test_loader, params.epochs)

    # Food172Model.predict(test_dataloaders=food172_test_loader)
    # 提取一个 validation batch, 在epoches 结束的时候查看预测结果。:::: 使用 ImagePredictionLogger
    # val_samples=next(iter(food172_test_loader))


    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
