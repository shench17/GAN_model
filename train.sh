CUDA_VISIBLE_DEVICES=gpu_id python main.py\
 --model_name corediff\
 --run_name dose5_mayo2016\
 --batch_size 4\
 --max_iter 150000\
 --test_dataset mayo_2016\
 --test_id 9\
 --context\
 --only_adjust_two_step\
 --dose 5\
 --save_freq 2500

