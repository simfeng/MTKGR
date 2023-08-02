import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def find_most_similar_vector(a, b):
    # 计算 a 和 b 之间的余弦相似度
    similarity_scores = F.cosine_similarity(a, b.unsqueeze(0), dim=1)
    # 找到最相似向量的索引
    most_similar_index = torch.argmax(similarity_scores)
    # 获取相似度值
    similarity_value = similarity_scores[most_similar_index]

    return most_similar_index.item(), similarity_value.item()


class AFIRE(object):
    def __init__(
        self,
        num_food,
        num_ingredients,
        train_food_label,
        train_ingreident_label,
        recipes=None,
        beta=1.0,
    ) -> None:
        """
        param:
            num_food: number of food
            num_ingredients: number of ingredients
            train_food_label: food label of training dataset, size: (num_data,)
            train_ingreident_label: ingredient label of training dataset, size: (num_data, num_ingredients)
        """
        print("====> init AFIRE <====")
        print(" \tnum_food:", num_food)
        print(" \tnum_ingredients:", num_ingredients)
        print(" \tbeta:", beta)
        super(AFIRE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.num_food = num_food
        self.num_ingredients = num_ingredients
        self.train_food_label = train_food_label
        self.train_ingredient_label = train_ingreident_label
        self.ingredient_proportion = []

        if recipes is None:
            self._ingredient_proportion()
        else:
            print("recipes:", len(recipes))
            self.ingredient_proportion = torch.tensor(recipes, device=self.device)

        self.result_record = []

    def set_beta(self, beta):
        print("==> set beta to:", beta)
        self.beta = beta

    def update_MovP(self, model, dataloader, save_movp=True, filename="MovP.pt"):
        """update moving precision of each food and each ingredient."""
        print("==> update MovP ...")
        self.MovP_file = filename
        self.save_movp = save_movp
        self.MovP_food = torch.zeros(self.num_food, device=self.device)
        self.MovP_ingredients = torch.zeros(self.num_ingredients, device=self.device)
        if os.path.isfile(self.MovP_file):
            print("==> load MovP from file ...")
            MovP = torch.load(self.MovP_file, map_location=self.device)

            self.MovP_food = MovP["MovP_food"]
            self.MovP_ingredient = MovP["MovP_ingredient"]
        else:
            self._update_MovP(model, dataloader)

    def _ingredient_proportion(self) -> None:
        """calculate ingredient proportion of each food in training dataset"""
        pd_food_label = pd.DataFrame(self.train_food_label)
        pd_ingredient_label = pd.DataFrame(self.train_ingredient_label)
        self.ingredient_proportion = []
        for i in range(self.num_food):
            # 使用索引属性找到值为目标值的索引
            result_indexes = pd_food_label[pd_food_label[0] == i].index.tolist()

            # 根据 result_indexes 找到对应的食材标签
            result_ingredient = pd_ingredient_label.loc[result_indexes]
            # print(result_ingredient.shape)
            # 统计每种食材的比例
            # result_ingredient 按每个位置求和
            result_ingredient_sum = result_ingredient.sum(axis=0)
            _proportion = result_ingredient_sum / len(result_indexes)
            self.ingredient_proportion.append(_proportion)

        self.ingredient_proportion = torch.tensor(
            self.ingredient_proportion, device=self.device
        )

        assert self.ingredient_proportion.shape == (self.num_food, self.num_ingredients)
        # 对于 food101 来讲，相同类别的食材是一致的，每种 food 的 ingredient_proportion 都是和菜谱一致的，里面只有 1 和 0
        print("ingredient_proportion:", self.ingredient_proportion)

    def _prepare_dataloader(self, dataloader):
        loader = [dataloader, iter(dataloader)]
        return loader, len(dataloader)

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])

        # data = data.to(self.device, non_blocking=True)
        # for task in self.task_name:
        #     label[task] = label[task].to(self.device, non_blocking=True)

        return data, label

    def _logits_to_preds(self, logits, topk=None):
        """transform logits to preds"""

        _logits_food = logits["multiclass"].softmax(1)
        if topk is None:
            pred_food = _logits_food.argmax(1).flatten().tolist()
        else:
            # 获取每行最大的topk个元素及其索引
            _, pred_food = torch.topk(_logits_food, k=topk, dim=1)
            pred_food = pred_food.cpu().numpy()

        pred_ingredients = (logits["multilabel"].sigmoid() > 0.5).float().tolist()
        return pred_food, pred_ingredients

    def _update_MovP(self, model, dataloader):
        """update moving precision of ech food and each ingredient.
        param:
            model: pretrained model
            test_loader: test dataloader

        pred_food = [1, 3, 92, 170, ....]
        pred_ingredients = [
            [1, 0, 1, ..., 0],
            [0, 1, 1, ..., 1],
            [0, 1, 1, ..., 1],
            [0, 1, 1, ..., 1],
            ...
        ]
        gt_food = [1, 2, 92, 171, ....],
        gt_ingredients = [
            [1, 0, 1, ..., 0],
            [0, 0, 1, ..., 1],
            [0, 1, 1, ..., 0],
            [0, 1, 1, ..., 1],
            ...
        ]
        pred_food[i] 表示第 i 个样本的预测 food 标签,
        pred_ingredients[i] 表示第 i 个样本的预测食材标签,
        gt_food[i] 表示第 i 个样本的真实 food 标签,
        gt_ingredients[i] 表示第 i 个样本的真实食材标签,
        """

        model.eval()

        pred_food = []
        pred_ingredients = []
        gt_food = []
        gt_ingredients = []

        dataloader, batch = self._prepare_dataloader(dataloader)

        with torch.no_grad():
            for batch_index in tqdm(range(batch)):
                _inputs, _gts = self._process_data(dataloader)
                # data_index = _gts["index"]
                logits = model(_inputs)
                # test_preds = self.process_preds(logits)
                _pred_food, _pred_ingredients = self._logits_to_preds(logits)

                pred_food.extend(_pred_food)
                pred_ingredients.extend(_pred_ingredients)
                gt_food.extend(_gts["multiclass"].flatten().tolist())

                # print('_gts["multiclass"].flatten().tolist():', _gts["multiclass"].flatten().tolist())
                gt_ingredients.extend(_gts["multilabel"].numpy())
                # print('len', len(gt_food), len(gt_ingredients))

        # transform element of results to pandas dataframe
        pred_food = torch.tensor(pred_food, device=self.device)
        pred_ingredients = torch.tensor(pred_ingredients, device=self.device)
        gt_food = torch.tensor(gt_food, device=self.device)
        gt_ingredients = torch.tensor(gt_ingredients, device=self.device)
        # print size of results
        print("==> pred_food size:", pred_food.shape)
        print("==> pred_ingredients size:", pred_ingredients.shape)
        print("==> gt_food size:", gt_food.shape)
        print("==> gt_ingredients size:", gt_ingredients.shape)

        print("Calc MovP_food ...")
        for i in range(self.num_food):
            # 找到预测为 i 的样本的索引
            result_indexes = (pred_food == i).nonzero()
            # 找到对应的 gt_food 标签
            result_gt_food = gt_food[result_indexes]
            # 计算 result_gt_food 中等于 i 的个数
            true_p = (result_gt_food == i).sum().item()
            all_p = len(result_indexes)
            self.MovP_food[i] = true_p / all_p if all_p else 0
            # print("=> i:", i, "true_p:", true_p, "all_p:", all_p, 'MovP:', self.MovP_food[i])

        # self.MovP_food = torch.tensor(self.MovP_food)
        print("==> MovP_food:", self.MovP_food.shape)

        print("Calc MovP_ingredient ...")
        for i in range(self.num_ingredients):
            # print(self.num_ingredients, i)
            # pred_ingredients是二维df，统计第二维中满足条件第 i 样本为 1 的索引
            result_indexes = []
            result_indexes = (pred_ingredients[:, i] == 1).nonzero()

            # 统计 gt_ingredients[j][i] == 1 的个数
            true_p = (gt_ingredients[result_indexes, i] == 1).sum().item()
            all_p = len(result_indexes)
            self.MovP_ingredients[i] = true_p / all_p if all_p else 0

        # self.MovP_ingredients = torch.tensor(self.MovP_ingredients)
        print("=> MovP_ingredients:", self.MovP_ingredients.shape)

        # get dir from self.MovP_file and check if dir exist
        if self.save_movp:
            print("==> save MovP to file:", self.MovP_file)
            _dir = os.path.dirname(self.MovP_file)
            if not os.path.isdir(_dir):
                os.makedirs(_dir)
            torch.save(
                {"MovP_food": self.MovP_food, "MovP_ingredient": self.MovP_ingredients},
                self.MovP_file,
            )

    def mpr_for_train(self, logits, ground_truth=None):
        """Moving Precision Ranking (MPR): use ingredient proportion and MovP to reason out the food and ingredient.
        param:
            logits: logits of test dataset,
                    {
                        "multiclass": torch.tensor(batch_size, num_food),
                        "multilabel": torch.tensor(batch_size, num_ingredients)
                    }
        return:
            labels: {
                "multiclass": torch.tensor(batch_size),
                "multilabel": torch.tensor(batch_size, num_ingredients)
            }
        """

        # print('----> groud_truth:', ground_truth['multiclass'].shape, ground_truth['multilabel'].shape)

        labels = {"multiclass": [], "multilabel": []}

        top5_food, ingredients = self._logits_to_preds(logits, topk=5)
        # top5_food size (batch_size, 5), ingredients size (batch_size, num_ingredients)

        logits_food = logits["multiclass"].softmax(1)  # size (batch_size, num_food)
        top1_food = logits_food.argmax(1).flatten().tolist()

        logits_ingredients = logits[
            "multilabel"
        ].sigmoid()  # size (batch_size, num_ingredients)

        batch_size = logits["multiclass"].size()[0]

        for i in range(batch_size):
            _food_label = ground_truth["multiclass"][i].item()
            # print('___food_label', _food_label)
            if _food_label != -1: # 如果有真实标签，则返回
                _ingredient_label = ground_truth["multilabel"][i].tolist()
                labels["multiclass"].append(_food_label)
                labels["multilabel"].append(_ingredient_label)
                continue
            
            # 没有真实标签，进行推理
            _top5_food = top5_food[i]  # 第 i 个样本的 top5 食物标签
            # print('--> _top5_food:', _top5_food, type(_top5_food))
            # _ingredients = ingredients[i]
            _logits_food = logits_food[i]
            _logits_ingredients = logits_ingredients[i]
            # find the most similiar with _logits_ingredients in self.ingredient_proportion
            _score = {}
            for _food_pred in _top5_food:
                # print("\n===============*****================")
                # print('device:\n')
                # print(self.ingredient_proportion.device, _logits_ingredients.device, self.MovP_ingredients.device)
                # print(_food_pred, type(_food_pred))
                # print(_logits_food[_food_pred].shape, type( _logits_food[_food_pred])) # torch.Size([]) <class 'torch.Tensor'>
                # print(_logits_ingredients.shape, type(_logits_ingredients)) # torch.Size([353]) <class 'torch.Tensor'>
                # print(self.ingredient_proportion[_food_pred].shape, type(self.ingredient_proportion[_food_pred])) # (172,) <class 'pandas.core.series.Series'>
                # print(self.MovP_ingredients.shape, type(self.MovP_ingredients))

                _food_score = (
                    _logits_food[_food_pred] * self.MovP_food[_food_pred]
                ).item()

                # print (_logits_ingredients > 0.5) 中 1 的个数
                # print('==> _logits_ingredients > 0.5:', (_logits_ingredients > 0.5).sum().item())

                # _ingredient_score = torch.sum(_logits_ingredients * (_logits_ingredients > 0.5) * self.MovP_ingredients).item()
                _ingredient_score = torch.sum(
                    (_logits_ingredients > 0.5)
                    * self.ingredient_proportion[_food_pred]
                    * self.MovP_ingredients
                ).item()
                # _ingredient_score = torch.sum(_logits_ingredients * self.ingredient_proportion[_food_pred] * self.MovP_ingredients).item()
                # _ingredient_score = torch.sum(_logits_ingredients * (_logits_ingredients > 0.5) * self.ingredient_proportion[_food_pred]).item()

                _score[_food_pred] = _food_score + self.beta * _ingredient_score
            # sort _score by value
            _score = sorted(_score.items(), key=lambda x: x[1], reverse=True)
            _food_label = _score[0][0]

            labels["multiclass"].append(_food_label)
            labels["multilabel"].append(
                self.ingredient_proportion[_food_label].tolist()
            )
        
        # return labels as tensor
        labels["multiclass"] = torch.tensor(labels["multiclass"], device=self.device)
        labels["multilabel"] = torch.tensor(labels["multilabel"], device=self.device)
        # print("==> labels:", labels["multiclass"].shape, labels["multilabel"].shape)
        return labels
    
    def mpr(self, logits, ground_truth=None):
        """Moving Precision Ranking (MPR): use ingredient proportion and MovP to reason out the food and ingredient.
        param:
            logits: logits of test dataset,
                    {
                        "multiclass": torch.tensor(batch_size, num_food),
                        "multilabel": torch.tensor(batch_size, num_ingredients)
                    }
        return:
            labels: {
                "multiclass": torch.tensor(batch_size),
                "multilabel": torch.tensor(batch_size, num_ingredients)
            }
        """

        labels = {"multiclass": [], "multilabel": []}

        top5_food, ingredients = self._logits_to_preds(logits, topk=5)
        # top5_food size (batch_size, 5), ingredients size (batch_size, num_ingredients)

        logits_food = logits["multiclass"].softmax(1)  # size (batch_size, num_food)
        top1_food = logits_food.argmax(1).flatten().tolist()

        logits_ingredients = logits[
            "multilabel"
        ].sigmoid()  # size (batch_size, num_ingredients)

        batch_size = logits["multiclass"].size()[0]

        for i in range(batch_size):
            _top5_food = top5_food[i]  # 第 i 个样本的 top5 食物标签
            # print('--> _top5_food:', _top5_food, type(_top5_food))
            # _ingredients = ingredients[i]
            _logits_food = logits_food[i]
            _logits_ingredients = logits_ingredients[i]
            # find the most similiar with _logits_ingredients in self.ingredient_proportion
            _food_label_by_ingredient, _similarity_value = find_most_similar_vector(
                self.ingredient_proportion, _logits_ingredients
            )
            _score = {}
            for _food_pred in _top5_food:
                # print("\n===============*****================")
                # print('device:\n')
                # print(self.ingredient_proportion.device, _logits_ingredients.device, self.MovP_ingredients.device)
                # print(_food_pred, type(_food_pred))
                # print(_logits_food[_food_pred].shape, type( _logits_food[_food_pred])) # torch.Size([]) <class 'torch.Tensor'>
                # print(_logits_ingredients.shape, type(_logits_ingredients)) # torch.Size([353]) <class 'torch.Tensor'>
                # print(self.ingredient_proportion[_food_pred].shape, type(self.ingredient_proportion[_food_pred])) # (172,) <class 'pandas.core.series.Series'>
                # print(self.MovP_ingredients.shape, type(self.MovP_ingredients))

                _food_score = (
                    _logits_food[_food_pred] * self.MovP_food[_food_pred]
                ).item()

                # print (_logits_ingredients > 0.5) 中 1 的个数
                # print('==> _logits_ingredients > 0.5:', (_logits_ingredients > 0.5).sum().item())

                # _ingredient_score = torch.sum(_logits_ingredients * (_logits_ingredients > 0.5) * self.MovP_ingredients).item()
                _ingredient_score = torch.sum(
                    (_logits_ingredients > 0.5)
                    * self.ingredient_proportion[_food_pred]
                    * self.MovP_ingredients
                ).item()
                # _ingredient_score = torch.sum(_logits_ingredients * self.ingredient_proportion[_food_pred] * self.MovP_ingredients).item()
                # _ingredient_score = torch.sum(_logits_ingredients * (_logits_ingredients > 0.5) * self.ingredient_proportion[_food_pred]).item()

                _score[_food_pred] = _food_score + self.beta * _ingredient_score
            # sort _score by value
            _score = sorted(_score.items(), key=lambda x: x[1], reverse=True)
            _food_label = _score[0][0]

            labels["multiclass"].append(_food_label)
            labels["multilabel"].append(
                self.ingredient_proportion[_food_label].tolist()
            )

            self.result_record.append(
                [
                    ground_truth["multiclass"][i].item(),
                    top1_food[i],
                    logits_food[i][top1_food[i]].item(),
                    _food_label,
                    _food_label_by_ingredient,
                    _similarity_value,
                ]
            )

        # labels["multiclass"] = torch.stack(labels["multiclass"])

        return labels
    
    def save_result_record(self, filename):
        print('==> save result_record to file:', filename)
        _dir = os.path.dirname(filename)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        columns = ["gt", "top1", "top1 logits", "mpr", "sim_food", "similarity"]
        df = pd.DataFrame(self.result_record, columns=columns)
        print("df size:", df.shape)
        df.to_csv(filename, index=False)
