# SOG
Self-Organizing Generator

```shell script
## Model Running Modes:

##=====MNIST=====

python3 train.py --gpu_ids 2,3,4,5 --dataset mnist --n_deconv 4 --n_conv 1 --n_latent 2 --n_rounds 1 --name mnist_deconv32_1conv --batch_size 48 --niter 10 --niter_decay 10 --save_epoch_freq 1
--continue_train


##======EMNIST======

python3 train.py --gpu_ids 0,1 --n_deconv 4 --n_latent 6 --n_rounds 3 --name emnist_conv64 --dataset emnist --batch_size 48 --display_freq 240

##======FMNIST======

if bugged >> set ngf = 128

python3 train.py --gpu_ids 4,5 --n_deconv 4  --n_conv 1 --n_latent 6 --n_rounds 3 --name fashion_mnist_conv64 --dataset fashion-mnist --batch_size 32 --niter 5 --niter_decay 5 --save_epoch_freq 1 --continue_train

##======FMNIST_VGG======

if bugged >> set ngf = 128
python3 train.py --gpu_ids 0,1 --n_deconv 4  --n_conv 1 --n_latent 6 --criterion vgg --lr 0.002 --n_rounds 3 --name fashion_mnist_conv64_vgg --dataset fashion-mnist --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 

##======EMNIST_ASYM======

python3 train.py --gpu_ids 0,1 --n_deconv 4 --n_conv 1 --n_latent 6 --n_rounds 3 --name emnist_conv64_asym --dataset emnist --batch_size 48 --display_freq 240 --criterion 'l1_asym' --match_criterion 'l1_asym'

##======CELEBA======

python3 train.py --gpu_ids 2,3,4,5 --n_deconv 5 --last_activation tanh --n_latent 8 --n_rounds 3 --name celeba --dataset celeba --dataroot /home/shared/datasets/celeba --img_size 64 --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 --nc 3  --display_freq 240 --lr 0.002


##======TABULAR======

python train.py --net_type  flat_mlp  --gpu_ids  2,3,4,5  --last_activation  none  --n_latent  6  --block_size  2  --samples_per_dim  30  --n_rounds  3  --name  power  --dataset  power  --niter  10  --niter_decay  10  --save_epoch_freq  1  --display_freq  2560  --save_latest_freq  20480  --lr  0.0001  --match_criterion  mse  --criterion  mse
python test.py --net_type  flat_mlp  --gpu_ids  2,3,4,5  --last_activation  none  --n_latent  6  --block_size  2  --samples_per_dim  30  --n_rounds  3  --name  power  --dataset  power  --match_criterion  mse  --criterion  mse

```
### Gym environments
For training/testing, run:
```shell script
python train.py --name gym --dataset gym --env_name Circles-v0 --net_type infogail_mlp --latent_optimizer ohs --n_latent 3 --gpu_ids 0,1 --criterion mse --match_criterion mse --print_freq 500 --save_latest_freq 64000 --save_epoch_freq 1 --niter 5 --niter_decay 5
python test.py --name gym --dataset gym --env_name Circles-v0 --net_type infogail_mlp --latent_optimizer ohs --n_latent 3 --gpu_ids 0,1 --match_criterion mse
```
Note that the argument `--gen_expert` is responsible for generating expert data. It is only necessary when you run `train` / `test` script for the first time. You might omit that next time you run the scripts since the expert file will already exist. 
