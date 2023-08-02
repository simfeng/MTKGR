
pretrain70:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 2 --scheduler step \
	--train_bs 32 --test_bs 32 --prefix labeled_70_ \
	--checkpoint_path "" \
	--action pretrain

test_afire_70:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 2 --scheduler step \
	--train_bs 32 --test_bs 64 --prefix '' \
	--checkpoint_path /www/logs_libmtl/food172_swin_t-labeled_70_-EW-HPS-202307312127/checkpoint_epoch_10.pth \
	--beta 0.913978495 --action test_afire

pretrain80:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix labeled_80_ \
	--checkpoint_path /www/logs_libmtl/food172_swin_t-labeled_80_-EW-HPS-202308011611/checkpoint_epoch_6.pth \
	--action pretrain

test_afire_80:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 2 --scheduler step \
	--train_bs 32 --test_bs 32 --prefix '' \
	--checkpoint_path  /www/logs_libmtl/food172_swin_t-labeled_80_-EW-HPS-202308011933/checkpoint_epoch_4.pth \
	--beta 0.913978495 --action test_afire

pretrain90:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 64 --test_bs 64 --prefix labeled_90_ \
	--checkpoint_path /www/logs_libmtl/food172_swin_t-labeled_90_-EW-HPS-202308010747/checkpoint_epoch_14.pth \
	--action pretrain

test_afire_90:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 0,1 --scheduler step \
	--train_bs 32 --test_bs 128 --prefix '' \
	--checkpoint_path  /www/logs_libmtl/food172_swin_t-labeled_90_-EW-HPS-202308010747/checkpoint_epoch_14.pth \
	--beta 0.913978495 --action test_afire

pretrain100:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 2,3 --scheduler step \
	--train_bs 64 --test_bs 64 --prefix "" \
	--checkpoint_path /www/logs_libmtl/food172_swin_t--EW-HPS-202307312128/checkpoint_epoch_14.pth \
	--action pretrain

test_afire_100:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Food172 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 32 --test_bs 256 --prefix '' \
	--checkpoint_path  /www/logs_libmtl/food172_swin_t--EW-HPS-202308011438/checkpoint_epoch_6.pth \
	--beta 0.913978495 --action test_afire