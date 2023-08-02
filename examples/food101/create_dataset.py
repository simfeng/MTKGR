from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Food101
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


class Food101(Dataset):
    """food-101 and ingredient-101 dataset for multi-task(image classification and multi-label classification) learning.
    ingredient-101 dataset file structure:
    Ingredients101/Annotations/
    |-- classes.txt # classes of food-101
    |-- ingredients_simplified.txt      # simplified ingredients of each class in food-101, we use this
    |-- ingredients.txt                 # ingredients of each class in food-101
    |-- recipesData_v1.json
    |-- recipesData_v2.json
    |-- test_images.txt                 # test images same as test.txt in food-101
    |-- test.json
    |-- test_labels.txt                 # labels of test images
    |-- test.txt
    |-- train_images.txt
    |-- train.json
    |-- train_labels.txt
    |-- train_short.txt
    |-- train_split.txt
    |-- train.txt
    |-- val_images.txt
    |-- val_labels.txt
    |-- val_split.txt
    `-- val.txt
    """

    num_classes = 101
    num_labels = 227

    def __init__(
        self,
        root,
        mode="train",
        prefix="",  # 'sub_' 表示只使用部分数据集
        sub_labels=[],  # 食材只使用部分标签
        augmentation=False,
        labeled_ratio=0.5, # 有标签数据的比例
        action='pretrain'
    ):
        print("=" * 20)
        print("root:", root)
        print("mode:", mode)
        print("prefix:", prefix)
        print("sub_labels:", sub_labels)
        print("augmentation:", augmentation)
        print("labeled_ratio:", labeled_ratio)
        print("action:", action)
        self.mode = mode
        self.labeled_ratio = labeled_ratio
        self.action = action
        self.sub_labels = sub_labels
        self.root = "/www/datasets/Ingredients101"
        self.image_dir = Path("/www/datasets/food-101") / "images"
        self.root = Path(self.root)
        self.label_dir = self.root / "Annotations"
        self.ingredient_label_path = self.label_dir / "ingredients_simplified.txt"
        self.augmentation = augmentation

        self.mode_map = {
            "train": f"{prefix}train_",
            "val": f"{prefix}val_",
            "trainval": [f"{prefix}train_", f"{prefix}val_"],
            "test": f"{prefix}test_",
        }
        assert (
            mode in self.mode_map.keys()
        ), f"split should be one of {self.mode_map.keys()}"

        self._init_transform()

        self._init_ingredient_label()

        self._init_food_label()

    def _init_food_label(self):
        # read TR.txt line by line save to self.images
        self.images = []
        self.multiclass = []  # 存储 label
        # print('image_dir', image_dir, type(image_dir), type(self.root))
        _mode = self.mode_map[self.mode]
        if isinstance(_mode, str):
            _mode = [_mode]

        # print('model:', _mode, self.mode, mode_map[mode])
        for _m in _mode:
            # load images
            with open(self.label_dir / f"{_m}images.txt", "r") as f:
                for line in f.readlines():
                    _line = line.rstrip()
                    _image_path = self.image_dir / f"{_line}.jpg"
                    assert os.path.isfile(_image_path), "{} not exist".format(
                        _image_path
                    )
                    self.images.append(_image_path)

            # load labels
            with open(self.label_dir / f"{_m}labels.txt", "r") as f:
                for line in f.readlines():
                    self.multiclass.append(int(line.rstrip()))
        
        if self.mode != 'test':
            self.multiclass = self._replace_with_percentage(
                np.array(self.multiclass), 
                1 - self.labeled_ratio)
            # print num of element equal to -1 and not equal to -1
            print('num of element equal to -1:', np.sum(np.array(self.multiclass) == -1))
            print('num of element not equal to -1:', np.sum(np.array(self.multiclass) != -1))
            # 删除 self.multiclass -1 的元素
            if 'pretrain' in self.action:
                self.images = [self.images[i] for i in range(len(self.multiclass)) if self.multiclass[i] != -1]
                self.multiclass = [self.multiclass[i] for i in range(len(self.multiclass)) if self.multiclass[i] != -1]



        assert len(self.multiclass) == len(self.images)

    def _insert_elements_interval(self, arr):
        # 计算后半部分的起始索引
        mid_index = len(arr) // 2
        # print(np.arange(mid_index, len(arr)), arr[mid_index:])

        # 将后半部分的元素间隔插入前半部分
        result = np.insert(arr[:mid_index], np.arange(0, len(arr[mid_index:])), arr[mid_index:])

        return result

    def _replace_with_percentage(self, data, percentage):
        unique_elements = np.unique(data)
        
        for element in unique_elements:
            indices = np.where(data == element)[0]
            sorted_indices = self._insert_elements_interval(np.sort(indices))
            # print('sorted_indices:', sorted_indices)
            # 将 sorted_indices 中的 第 len(sorted_indices)/2 个元素放在第 0 个元素后面，
            num_to_replace = int(len(indices) * percentage)
            indices_to_replace = sorted_indices[:num_to_replace]
            data[indices_to_replace] = -1
        
        # return data as list
        return data.tolist()


    def _init_ingredient_label(self):
        # read ingredient label
        # 读取标签，弄成 all hot 的形式, 参考 food 172
        self.recipes_map = {}
        self.recipes = [[]] * Food101.num_classes
        self.ingredients = set()

        with open(self.ingredient_label_path, "r") as f:
            i = 0
            for line in f.readlines():
                # Remove leading and trailing whitespaces, then split by comma
                _ingredients = line.strip().split(",")
                # Convert the list to a set to remove duplicates, then back to a list
                self.ingredients.update(set(_ingredients))
                self.recipes_map[i] = set(_ingredients)
                i += 1

        Food101.num_labels = len(self.ingredients)

        for _food_label, _ingredients in self.recipes_map.items():
            self.recipes[_food_label] = [
                1 if _i in _ingredients else 0 for _i in self.ingredients
            ]
            # print(_food_label, self.recipes[_food_label])

        # if self.sub_labels:
        #     Food101.num_labels = len(self.sub_labels)
        #     for _k, _v in self.recipes_map.items():
        #         self.recipes[_k] = [_v[i] for i in self.sub_labels]

        print("num_labels:", Food101.num_labels)

    def __getitem__(self, index):
        image = self._load_img(index)

        image = self.train_transformers(image)
        food_label = self.multiclass[index]
        if food_label == -1:
            ingredient_label = torch.tensor([-1] * Food101.num_labels)
        else:
            ingredient_label = torch.tensor(self.recipes[food_label])
        return image, {"multiclass": food_label, "multilabel": ingredient_label}

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        # resize to 256
        _img = _img.resize((256, 256))
        # _img = np.array(_img, dtype=np.float32, copy=False)
        return _img

    def _init_transform(self):
        if self.augmentation:
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

    def __repr__(self):
        return self.__class__.__name__ + "()"
