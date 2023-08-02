from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

import os
import torch
import torch.nn.functional as F
import random
from PIL import Image
from pathlib import Path
import pandas as pd


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """

    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(
            img[None, :, i : i + h, j : j + w],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        return img_


class Food172(Dataset):
    """food172 dataset for multi-task(image classification and multi-label classification) learning.
    Food172 dataset file structure:
    Fodd172/
    ├── SplitAndIngreLabel
    │   ├── FoodList.txt
    │   ├── IngreLabel.txt
    │   ├── IngredientList.txt
    │   ├── TE.txt
    │   ├── TR.txt
    │   └── VAL.txt
    └── ready_chinese_food
        ├── 1
        ├── 10
        ...
        └── 99
    """

    num_classes = 172
    num_labels = 353

    def __init__(
        self,
        root,
        mode="train",
        prefix="",  # 'sub_' 表示只使用部分数据集
        sub_labels=[],  # 食材只使用部分标签
        augmentation=False,
    ):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        ### sdf: transform raw image
        self.transform = augmentation
        if self.transform:
            # 定义数据增强的操作
            self.train_transformers = transforms.Compose(
                [
                    transforms.RandomResizedCrop(256),  # 随机裁剪
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                    ),  # 随机颜色抖动
                    transforms.RandomRotation(degrees=15),  # 随机旋转
                    transforms.ToTensor(),  # 转换为张量
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # 归一化
                ]
            )

        else:
            self.train_transformers = transforms.Compose(
                [
                    transforms.ToTensor(),  # 转换为张量
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # 归一化
                ]
            )

        mode_map = {
            "train": f"{prefix}TR.txt",
            "val": f"{prefix}VAL.txt",
            "trainval": [f"{prefix}TR.txt", f"{prefix}VAL.txt"],
            "test": f"TE.txt",
        }
        assert mode in mode_map.keys(), "split should be one of {}".format(
            mode_map.keys()
        )

        self.root = Path(self.root)
        self.label_dir = self.root / "SplitAndIngreLabel"
        self.image_dir = self.root / "ready_chinese_food"
        self.food_label_path = self.label_dir / "FoodList.txt"
        self.ingredient_label_path = self.label_dir / "IngreLabel.txt"
        self.ingredient_ingredient_path = self.label_dir / "IngredientList.txt"

        self.image_ingredient_label_map = {}
        # read ingredient label
        with open(self.ingredient_label_path, "r") as f:
            for line in f.readlines():
                _l = line.strip().split(" ")
                self.image_ingredient_label_map[_l[0]] = list(
                    map(lambda x: max(0, int(x)), _l[1:])
                )
                # print(_l[0], len(self.image_ingredient_label_map[_l[0]]), self.image_ingredient_label_map[_l[0]])
                # assert False

        if sub_labels:
            Food172.num_labels = len(sub_labels)
            for _k, _v in self.image_ingredient_label_map.items():
                self.image_ingredient_label_map[_k] = [_v[i] for i in sub_labels]

        # read FoodList.txt line by line save to self.food_label_names
        self.food_label_names = []
        with open(self.food_label_path, "r") as f:
            for line in f.readlines():
                _l = line.strip()
                self.food_label_names.append(_l)

        # ingredient label names
        self.ingredient_label_names = []
        with open(self.ingredient_ingredient_path, "r") as f:
            for line in f.readlines():
                _l = line.strip()
                self.ingredient_label_names.append(_l)

        # read TR.txt line by line save to self.images
        self.images = []
        self.multiclass = []  # 存储 label
        self.multilabel = []
        # print('image_dir', image_dir, type(image_dir), type(self.root))
        _mode = mode_map[mode]
        if isinstance(_mode, str):
            _mode = [_mode]
        # print('model:', _mode, self.mode, mode_map[mode])
        for _m in _mode:
            with open(self.label_dir / _m, "r") as f:
                for line in f.readlines():
                    _line = line.rstrip()
                    _image_path = self.image_dir / _line[1:]
                    assert os.path.isfile(_image_path), "{} not exist".format(
                        _image_path
                    )
                    self.images.append(_image_path)
                    _label = _line.split("/")[1]  # label start from 1
                    self.multiclass.append(int(_label) - 1)
                    self.multilabel.append(self.image_ingredient_label_map[_line])

        assert len(self.multiclass) == len(self.images)
        
        self._generate_recipe()

    def _generate_recipe(self):
        """calculate ingredient proportion of each food in training dataset"""
        food_label = []
        ingredient_label = []
        files = ["VAL.txt", "TR.txt", "TE.txt"]
        for file in files:
            with open(self.root / "SplitAndIngreLabel" / file, "r") as f:
                for line in f.readlines():
                    _line = line.rstrip()
                    _label = _line.split("/")[1]
                    food_label.append(int(_label) - 1)
                    ingredient_label.append(self.image_ingredient_label_map[_line])


        pd_food_label = pd.DataFrame(food_label)
        pd_ingredient_label = pd.DataFrame(ingredient_label)
        self.recipes = []
        for i in range(self.num_classes):
            # 使用索引属性找到值为目标值的索引
            result_indexes = pd_food_label[pd_food_label[0] == i].index.tolist()

            # 根据 result_indexes 找到对应的食材标签
            result_ingredient = pd_ingredient_label.loc[result_indexes]
            # print(result_ingredient.shape)
            # 统计每种食材的比例
            # result_ingredient 按每个位置求和
            result_ingredient_sum = result_ingredient.sum(axis=0)
            _proportion = result_ingredient_sum / len(result_indexes)
            self.recipes.append(_proportion)

        self.recipes = torch.tensor(self.recipes)

        self.recipes = (self.recipes > 0.5).int()

        assert self.recipes.shape == (self.num_classes, self.num_labels)
        # print self.recipes 中每一行的和
        print(self.recipes.sum(axis=1))

        self.recipes = self.recipes.tolist()

    def __getitem__(self, index):

        image = self._load_img(index)

        image = self.train_transformers(image)

        return image, {
            "multiclass": self.multiclass[index],
            "multilabel": torch.tensor(self.recipes[self.multiclass[index]]),
        }

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        # _img = np.array(_img, dtype=np.float32, copy=False)
        return _img

    def __repr__(self):
        return self.__class__.__name__ + "()"
