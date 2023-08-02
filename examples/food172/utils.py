import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss

class MultiLabelMetric(AbsMetric):
    def __init__(self):
        super(MultiLabelMetric, self).__init__()

        self.micro_precision = []
        self.micro_recall = []
        self.micro_f1 = []
        self.macro_precision = []
        self.macro_recall = []
        self.macro_f1 = []
        self.samples_precision = []
        self.samples_recall = []
        self.samples_f1 = []
        self.bs = []
            
        
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        
        if not isinstance(pred, list):
            pred = pred.sigmoid()
            pred = (pred > 0.5).float()
            pred_numpy = pred.cpu().detach().numpy()
            batch_size = pred.size()[0]
        else:
            # list to numpy
            pred_numpy = np.array(pred)
            batch_size = len(pred)

        self.bs.append(batch_size)
        gt_numpy = gt.cpu().detach().numpy()
            
        self.micro_precision.append(precision_score(y_true=gt_numpy, y_pred=pred_numpy, average='micro', zero_division=1))
        self.micro_recall.append(recall_score(y_true=gt_numpy, y_pred=pred_numpy, average='micro', zero_division=1))
        self.micro_f1.append(f1_score(y_true=gt_numpy, y_pred=pred_numpy, average='micro', zero_division=1))
        self.macro_precision.append(precision_score(y_true=gt_numpy, y_pred=pred_numpy, average='macro', zero_division=1))
        self.macro_recall.append(recall_score(y_true=gt_numpy, y_pred=pred_numpy, average='macro', zero_division=1))
        self.macro_f1.append(f1_score(y_true=gt_numpy, y_pred=pred_numpy, average='macro', zero_division=1))
        self.samples_precision.append(precision_score(y_true=gt_numpy, y_pred=pred_numpy, average='samples', zero_division=1))
        self.samples_recall.append(recall_score(y_true=gt_numpy, y_pred=pred_numpy, average='samples', zero_division=1))
        self.samples_f1.append(f1_score(y_true=gt_numpy, y_pred=pred_numpy, average='samples', zero_division=1))

        

    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        return [np.mean(self.micro_precision), np.mean(self.micro_recall), 
                np.mean(self.micro_f1), np.mean(self.macro_precision), 
                np.mean(self.macro_recall), np.mean(self.macro_f1),
                np.mean(self.samples_precision), np.mean(self.samples_recall), 
                np.mean(self.samples_f1)]
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.micro_precision = []
        self.micro_recall = []
        self.micro_f1 = []
        self.macro_precision = []
        self.macro_recall = []
        self.macro_f1 = []
        self.samples_precision = []
        self.samples_recall = []
        self.samples_f1 = []
        self.bs = []

class MultiLabelLoss(AbsLoss):
    """
    y_true = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]],dtype=torch.int16)
    y_pred = torch.tensor([[0.2, 0.5, 0, 0], [0.1, 0.5, 0, 0.8]],dtype=torch.float32)
    loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
    print(loss(y_pred, y_true)) #0.5926
    """
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss(reduction='mean')
        # self.loss_fn = nn.BCELoss()
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())

# multiclass
class MultiClassMetric(AbsMetric):
    def __init__(self):
        super(MultiClassMetric, self).__init__()

        self.top_1_record = []
        self.top_5_record = []
        
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        # print('pred:', pred, type(pred))
        if isinstance(pred, list):
            # 已经计算好结果的
            self.top_1_record.append(self._get_top_1_by_revised_label(pred, gt))
            batch_size = len(pred)
        else:
            self.top_1_record.append(self._get_top_1(pred, gt))
            self.top_5_record.append(self._get_top_5(pred, gt))
            batch_size = pred.size()[0]
        self.bs.append(batch_size)
    
    def _get_top_1_by_revised_label(self, revised_label, gt):
        # revised_label: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # revised_label to tensor
        revised_label = torch.tensor(revised_label, dtype=torch.long).to(gt.device)
        gt = gt.long().flatten()
        acc = (revised_label == gt).sum().item() / revised_label.size()[0]
        return acc
        
    def _get_top_1(self, pred, gt):
        pred = pred.softmax(1).argmax(1).flatten()
        gt = gt.long().flatten()
        acc = (pred == gt).sum().item() / pred.size()[0]
        return acc
    
    def _get_top_5(self, pred, gt):
        pred = pred.softmax(1)
        gt = gt.long().flatten()
        _, top5 = torch.topk(pred, 5, dim=1)
        acc = (top5 == gt.unsqueeze(1)).sum().item() / pred.size()[0]
        return acc

    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        return [np.mean(self.top_1_record), np.mean(self.top_5_record)]
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.top_1_record = []
        self.top_5_record = []
        self.bs = []

class MultiClassLoss(AbsLoss):
    def __init__(self):
        super(MultiClassLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, pred, gt):
        # cross_entropy 不需要经过 softmax
        return self.loss_fn(pred, gt.long())

