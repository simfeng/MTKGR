import sys
import wandb
from datetime import datetime
from LibMTL.utils import set_device
from LibMTL.config import LibMTL_args
from examples.food101.train_food101_swin_t import prepare_dataloader, prepare_model

sys.path.append('../..')
from afire import AFIRE

wandb.login(key="<your-wandb-key>")
wandb.init(project="AFIRE_Food101", dir='/logs/wandb')

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=True, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=16, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--prefix', default='', type=str, help='')
    parser.add_argument('--labeled_ratio', default=0.5, type=float, help='')
    parser.add_argument('--sub_labels', default='', type=str, help='')
    parser.add_argument('--checkpoint_path', default='', type=str, help='')
    parser.add_argument('--beta', default=0.9, type=float, help='')
    parser.add_argument('--action', default='test_afire', type=str, help='')
    return parser.parse_args()

def _get_save_checkpoint_path(params):

    # 获取当前日期和时间
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d%H%M") 

    # 按时间保存模型
    save_checkpoint_path = f'/www/logs_libmtl/food101_swin_t-{params.prefix}-{params.weighting}-{params.arch}-{date_time_str}'
    
    return save_checkpoint_path


def pretrain(params):
    train_loader, test_loader = prepare_dataloader(params)
    save_checkpoint_path = _get_save_checkpoint_path(params)

    print('len of train_loader:', len(train_loader))
    print('len of test_loader:', len(test_loader))
    Food101Model = prepare_model(params, 
                                 save_checkpoint_path=save_checkpoint_path)
    Food101Model.train(train_loader, test_loader, params.epochs)


def afire(params):
    train_loader, test_loader = prepare_dataloader(params)
    save_checkpoint_path = _get_save_checkpoint_path(params)
    Food101Model = prepare_model(params, 
                                 save_checkpoint_path=save_checkpoint_path)
    # Food101Model.train(train_loader, test_loader, params.epochs)
    print('len of train_loader:', len(train_loader))
    print('len of test_loader:', len(test_loader))
    train_dataset = train_loader.dataset
    afire = AFIRE(num_food=train_dataset.num_classes, 
                  num_ingredients=train_dataset.num_labels,
                  train_food_label=train_dataset.multiclass,
                  train_ingreident_label=None,
                  recipes=train_dataset.recipes,
                  beta=params.beta)
    
    afire.update_MovP(Food101Model.model, test_loader, save_movp=False,
                      filename=f'{params.labeled_ratio or "all"}_mix_MovP.pt')
    # Food101Model = nn.DataParallel(Food101Model)
    Food101Model.train(train_loader, test_loader, params.epochs, afire=afire)

def test_acc(params):
    """计算测试集准确率
    """
    Food101Model = prepare_model(params)
    _, food101_test_loader = prepare_dataloader(params)

    Food101Model.test(test_dataloaders=food101_test_loader)
    # 提取一个 validation batch, 在epoches 结束的时候查看预测结果。:::: 使用 ImagePredictionLogger
    # val_samples=next(iter(food101_test_loader))

def test_afire(params):
    save_checkpoint_path = _get_save_checkpoint_path(params)
    
    train_loader, test_loader = prepare_dataloader(params)
    
    Food101Model = prepare_model(params, save_checkpoint_path)

    # Food101Model.test(test_dataloaders=test_loader, epoch=0)

    train_dataset = train_loader.dataset
    afire = AFIRE(num_food=train_dataset.num_classes, 
                  num_ingredients=train_dataset.num_labels,
                  train_food_label=train_dataset.multiclass,
                  train_ingreident_label=None,
                  recipes=train_dataset.recipes,
                  beta=params.beta)
    
    afire.update_MovP(Food101Model.model, test_loader, 
                      filename=f'./{params.labeled_ratio or "all_"}_mix_test_MovP.pt')
    epoch = 0
    # for beta in [0, 0.1, 0.2, 0.3, 0.4]:
    for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]:
        afire.set_beta(beta)
        Food101Model.test(test_dataloaders=test_loader, 
                          epoch=epoch, afire=afire)
        afire.save_result_record(f'./{params.labeled_ratio or "all_"}{beta}_afire_result.csv')
        epoch += 1
    
    # save afire.result_record to csv
    # afire.save_result_record(f'/www/logs_libmtl/afire/{params.prefix or "all_"}afire.csv')
    

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
    elif params.action == 'afire':
        afire(params)
