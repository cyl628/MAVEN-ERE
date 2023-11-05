python -u main.py --epochs 50 --log_steps 20 --eval_steps 300 --lr 3e-5 --model_name /data/MODEL/flan-t5-xxl --batch_size 1 &
# CUDA_VISIBLE_DEVICES=2,3 python -u main.py --epochs 50 --log_steps 20 --eval_steps 100 --lr 3e-5 --model_name /data/MODEL/flan-t5-large --batch_size 8 --eval_only &
wait
