import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from pprint import pprint

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters


class Trainer(nn.Module):
    r"""A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    """

    def __init__(
        self,
        task_dict,
        weighting,
        architecture,
        encoder_class,
        decoders,
        rep_grad,
        multi_input,
        optim_param,
        scheduler_param,
        checkpoint_path,
        save_checkpoint_path=None,
        **kwargs,
    ):
        super(Trainer, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.checkpoint_path = checkpoint_path
        self.save_checkpoint_path = save_checkpoint_path

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)

        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        class MTLmodel(architecture, weighting):
            def __init__(
                self,
                task_name,
                encoder_class,
                decoders,
                rep_grad,
                multi_input,
                device,
                kwargs,
            ):
                super(MTLmodel, self).__init__(
                    task_name,
                    encoder_class,
                    decoders,
                    rep_grad,
                    multi_input,
                    device,
                    **kwargs,
                )
                self.init_param()

        self.model = MTLmodel(
            task_name=self.task_name,
            encoder_class=encoder_class,
            decoders=decoders,
            rep_grad=self.rep_grad,
            multi_input=self.multi_input,
            device=self.device,
            kwargs=self.kwargs["arch_args"],
        ).to(self.device)
        # self.dp_model = nn.DataParallel(self.model)
        # self.model = self.dp_model.module
        count_parameters(self.model)

        if os.path.isfile(self.checkpoint_path):
            print("loading checkpoint {}".format(self.checkpoint_path))
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print("=====loading done!=====")

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
        }
        scheduler_dict = {
            "exp": torch.optim.lr_scheduler.ExponentialLR,
            "step": torch.optim.lr_scheduler.StepLR,
            "cos": torch.optim.lr_scheduler.CosineAnnealingLR,
            "reduce": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        optim_arg = {k: v for k, v in optim_param.items() if k != "optim"}
        self.optimizer = optim_dict[optim_param["optim"]](
            self.model.parameters(), **optim_arg
        )
        if scheduler_param is not None:
            scheduler_arg = {
                k: v for k, v in scheduler_param.items() if k != "scheduler"
            }
            self.scheduler = scheduler_dict[scheduler_param["scheduler"]](
                self.optimizer, **scheduler_arg
            )
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def process_preds(self, preds, task_name=None):
        r"""The processing of prediction for each task.

        - The default is no processing. If necessary, you can rewrite this function.
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        """
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        train_losses = torch.zeros(self.task_num).to(self.device)
        for tn, task in enumerate(self.task_name):
            train_losses[tn] = self.meter.losses[task]._update_loss(
                preds[task], gts[task]
            )
        return train_losses

    def _prepare_dataloaders(self, dataloaders):
        loader = [dataloaders, iter(dataloaders)]
        return loader, len(dataloaders)

    def on_validation_epoch_end(self, images, logits, labels):
        # val_images,val_labels = val_samples

        # 获取验证结果:
        # logits=Food172Model.model(val_images)
        multilabel_label = labels["multilabel"]
        multiclass_label = labels["multiclass"]
        multilabel_logits = logits["multilabel"]
        multiclass_logits = logits["multiclass"]
        # 将 val_samples tensor 导入cpu
        images = images.to(device=self.device)
        multilabel_label = multilabel_label.to(device=self.device)
        multiclass_label = multiclass_label.to(device=self.device)

        if isinstance(multiclass_logits, list):  # logits 就是预测的结果
            multiclass_preds = multiclass_logits
            multilabel_preds = multilabel_logits
        else:
            multiclass_preds = torch.argmax(multiclass_logits, dim=1)
            multilabel_preds = [
                [int(val >= 0.5) for val in row] for row in multilabel_logits.sigmoid()
            ]

        # 将图片 log 到 wandb image
        exampes = []
        for i in range(len(images)):
            _m_label = [i for i, x in enumerate(multilabel_label[i].tolist()) if x == 1]
            _m_pred = [i for i, x in enumerate(multilabel_preds[i]) if x == 1]
            exampes.append(
                wandb.Image(
                    images[i],
                    caption=f"{multiclass_label[i]}: {multiclass_preds[i]}, {_m_label}: {_m_pred}",
                )
            )
        wandb.log({"example_image": exampes})

    def train(
        self,
        train_dataloaders,
        test_dataloaders,
        epochs,
        afire=None,
        val_dataloaders=None,
        return_weight=False,
    ):
        r"""The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        """
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time("begin")
            for batch_index in tqdm(range(train_batch)):
                train_inputs, train_gts = self._process_data(train_loader)
                train_preds = self.model(train_inputs)
                # print(train_preds['multilabel'].sigmoid().tolist())
                train_preds = self.process_preds(train_preds)
                if afire:
                    train_gts = afire.mpr_for_train(train_preds, train_gts)

                train_losses = self._compute_loss(train_preds, train_gts)
                self.meter.update(train_preds, train_gts)

                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs["weight_args"])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

            self.meter.record_time("end")
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            log_dict = self.meter.display(epoch=epoch, mode="train")
            wandb.log(log_dict)
            self.meter.reinit()

            
            if afire and epoch % 5 == 0:
                print("==> update movp ...")
                afire.update_MovP(
                    self.model,
                    test_dataloaders,
                    filename=os.path.join(
                        self.save_checkpoint_path, f"{epoch}_MovP.pt"
                    ),
                )
            # 每隔 n 个epoch保存一次checkpoint
            if self.save_checkpoint_path and epoch % 2 == 0:
                if not os.path.exists(self.save_checkpoint_path):
                    os.makedirs(self.save_checkpoint_path)
                save_to = os.path.join(
                    self.save_checkpoint_path, f"checkpoint_epoch_{epoch}.pth"
                )
                print("\nsaving checkpoint to {}".format(save_to))
                torch.save(self.model.state_dict(), save_to)

            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(
                    val_dataloaders, epoch, mode="val", return_improvement=True
                )

            self.test(test_dataloaders, epoch, mode="test", afire=afire)
            if self.scheduler is not None:
                if (
                    self.scheduler_param["scheduler"] == "reduce"
                    and val_dataloaders is not None
                ):
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def test(
        self,
        test_dataloaders,
        epoch=None,
        mode="test",
        return_improvement=False,
        afire=None,
    ):
        r"""The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time("begin")
        with torch.no_grad():
            for batch_index in tqdm(range(test_batch)):
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds(test_preds)
                test_losses = self._compute_loss(test_preds, test_gts)
                if afire:
                    test_preds = afire.mpr(test_preds, test_gts)
                self.meter.update(test_preds, test_gts)

        self.meter.record_time("end")
        self.meter.get_score()
        epoch = 0 if epoch is None else epoch
        log_dict = self.meter.display(epoch=epoch, mode=mode)
        wandb.log(log_dict)
        # self.on_validation_epoch_end(
        #     images=test_inputs, logits=test_preds, labels=test_gts
        # )
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement

    def predict(self, test_dataloaders, save_to):
        r"""The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
        """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time("begin")
        results = []
        with torch.no_grad():
            for batch_index in range(test_batch):
                test_inputs, test_gts = self._process_data(test_loader)
                data_index = test_gts["index"]
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds(test_preds)
                results.append(
                    {
                        "index": data_index,
                        "multiclass": test_preds["multiclass"].tolist(),
                        "multilabel": test_preds["multilabel"].sigmoid().tolist(),
                    }
                )

        # load results to pandas dataframe and save to csv
        results = pd.DataFrame(results)
        results.to_csv(save_to, index=False)
