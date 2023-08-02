import sys
import wandb
from LibMTL.utils import set_device
from LibMTL.config import LibMTL_args
from train_food172_swin_t import prepare_dataloader, prepare_model

sys.path.append('../..')
from afire import AFIRE

wandb.login(key="<your-wandb-key>")
wandb.init(project="AFIRE_Food172", dir='/logs/wandb')

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=True, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=16, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--prefix', default='', type=str, help='')
    parser.add_argument('--sub_labels', default='', type=str, help='')
    parser.add_argument('--checkpoint_path', default='', type=str, help='')
    parser.add_argument('--beta', default=0.9, type=float, help='')
    parser.add_argument('--action', default='test_afire', type=str, help='')
    return parser.parse_args()

def pretrain(params):
    Food172Model = prepare_model(params)
    train_loader, test_loader = prepare_dataloader(params)
    print("train len:", len(train_loader), "test len:", len(test_loader))

    # Food172Model = nn.DataParallel(Food172Model)
    Food172Model.train(train_loader, test_loader, params.epochs)


def test_acc(params):
    """计算测试集准确率
    """
    Food172Model = prepare_model(params)
    _, food172_test_loader = prepare_dataloader(params)

    Food172Model.test(test_dataloaders=food172_test_loader)
    # 提取一个 validation batch, 在epoches 结束的时候查看预测结果。:::: 使用 ImagePredictionLogger
    # val_samples=next(iter(food172_test_loader))

def test_afire(params):

    Food172Model = prepare_model(params)
    train_loader, test_loader = prepare_dataloader(params)

    train_dataset = train_loader.dataset
    afire = AFIRE(num_food=train_dataset.num_classes, 
                  num_ingredients=train_dataset.num_labels,
                  train_food_label=train_dataset.multiclass,
                  train_ingreident_label=train_dataset.multilabel,
                  beta=params.beta)
    
    afire.update_MovP(Food172Model.model, test_loader, 
                      filename=f'./{params.prefix or "all"}_mix_MovP.pt')
    
    for i, beta in enumerate([0, 0.1, 0.2, 0.3]):
        afire.set_beta(beta)
        Food172Model.test(test_dataloaders=test_loader, 
                          epoch=i, afire=afire)

if __name__ == '__main__':
    # pretrain(params)
    # test_afire()
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)

    if params.action == 'pretrain':
        pretrain(params)
    elif params.action == 'test_acc':
        test_acc(params)
    elif params.action == 'test_afire':
        test_afire(params)
