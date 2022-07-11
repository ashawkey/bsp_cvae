#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python main.py --checkpoint latest --test_shapenet
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python main.py --checkpoint latest --interpolated_generate
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python main.py --checkpoint latest --generate
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python main.py --checkpoint latest --save_z
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python main.py --checkpoint latest --save_db
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python main.py --checkpoint latest --test_scannet
