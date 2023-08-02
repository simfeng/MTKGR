afire_mix_50:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" --labeled_ratio 0.5 \
	--checkpoint_path '' \
	--beta 0.3 --action afire

pretrain50:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" --labeled_ratio 0.5 \
	--action pretrain

pretrain80:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" --labeled_ratio 0.8 \
	--action pretrain
test_afire80:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 256 --prefix "" \
	--checkpoint_path /www/logs_libmtl/food101_swin_t--EW-HPS-202307311453/checkpoint_epoch_6.pth \
	--action test_afire

pretrain70:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1 --scheduler step \
	--train_bs 64 --test_bs 64 --prefix "" --labeled_ratio 0.7 \
	--action pretrain
test_afire70:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" \
	--checkpoint_path /www/logs_libmtl/food101_swin_t--EW-HPS-202307311943/checkpoint_epoch_24.pth \
	--action test_afire

pretrain90:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" --labeled_ratio 0.9 \
	--action pretrain

afire_mix_90:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 128 --prefix "" --labeled_ratio 0.9 \
	--checkpoint_path /www/logs_libmtl/food101_swin_t--EW-HPS-202307310907/checkpoint_epoch_10.pth \
	--beta 0.3 --action afire

test_afire90:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 128 --test_bs 256 --prefix "" \
	--checkpoint_path /www/logs_libmtl/food101_swin_t--EW-HPS-202307310907/checkpoint_epoch_10.pth \
	--action test_afire

test_afire100:
	python main.py --weighting EW --arch HPS \
	--dataset_path /www/datasets/Ingredient-101 --gpu_id 0,1,2,3 --scheduler step \
	--train_bs 32 --test_bs 256 --prefix "" \
	--checkpoint_path /www/logs_libmtl/food101_swin_t--EW-HPS-202307301413/checkpoint_epoch_13.pth \
	--beta 0.913978495 --action test_afire
