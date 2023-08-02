from LibMTL import Trainer

checkpoint_path = 'checkpoints/checkpoint_epoch_3.pth'

model = Trainer.load_from_checkpoint(checkpoint_path)


print(model)