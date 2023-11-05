# CUDA_VISIBLE_DEVICES=3 python -u main.py --epochs 50 --log_steps 50 --eval_steps 300 --lr 3e-5 --model_name /data/MODEL/flan-t5-small --batch_size 8 &
# CUDA_VISIBLE_DEVICES=1 python -u main.py --epochs 50 --log_steps 50 --eval_steps 300 --lr 3e-5 --model_name /data/MODEL/flan-t5-large --batch_size 8 --eval_only &
# python -u main.py --epochs 50 --log_steps 20 --eval_steps 100 --lr 3e-5 --model_name /data/MODEL/flan-t5-xxl --batch_size 1 --ddp &
torchrun --nnodes 1 --nproc_per_node 4  main.py --epochs 50 --log_steps 20 --eval_steps 100 --lr 3e-5 --model_name /data/MODEL/flan-t5-xxl --batch_size 1 --ddp
# python -u  main.py --epochs 50 --log_steps 20 --eval_steps 100 --lr 3e-5 --model_name /data/MODEL/flan-t5-xxl --batch_size 1


# wait
