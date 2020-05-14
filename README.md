# SOG
Self-Organizing Generator

## Model Running Modes:

##=====MNIST=====

--gpu_ids 2,3,4,5 --dataset mnist --n_deconv 4 --n_conv 1 --n_latent 2 --n_rounds 1 --name mnist_deconv32_1conv --batch_size 48 --niter 10 --niter_decay 10 --save_epoch_freq 1
--continue_train


##======EMNIST======

/home/arash/anaconda3/bin/python -u /home/arash/Desktop/SOG/train.py --gpu_ids 0,1 --n_deconv 4 --n_latent 6 --n_rounds 3 --name emnist_conv64 --dataset emnist --batch_size 48 --display_freq 240

##======FMNIST======

if bugged >> set ngf = 128

--gpu_ids 4,5 --n_deconv 4  --n_conv 1 --n_latent 6 --n_rounds 3 --name fashion_mnist_conv64 --dataset fashion-mnist --batch_size 32 --niter 5 --niter_decay 5 --save_epoch_freq 1 --continue_train

##======FMNIST_VGG======

if bugged >> set ngf = 128
--gpu_ids 0,1 --n_deconv 4  --n_conv 1 --n_latent 6 --criterion vgg --lr 0.002 --n_rounds 3 --name fashion_mnist_conv64_vgg --dataset fashion-mnist --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 

##======EMNIST_ASYM======

/home/arash/anaconda3/bin/python -u /home/arash/Desktop/SOG/train.py --gpu_ids 0,1 --n_deconv 4 --n_conv 1 --n_latent 6 --n_rounds 3 --name emnist_conv64_asym --dataset emnist --batch_size 48 --display_freq 240 --criterion 'l1_asym' --match_criterion 'l1_asym'

##======CELEBA-LBFGS-======

fix_adam>>--gpu_ids 0,1 --n_deconv 5 --last_activation tanh --n_latent 10 --name celeba_adam --dataset celeba --dataroot /home/shared/datasets/celeba --latent_optimizer lbfgs --num_lbfgs_steps 100 --criterion vgg --img_size 64 --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 --nc 3  --display_freq 240 --lr 0.002 

fix_lbfgs>>--gpu_ids 0,1 --n_deconv 5 --last_activation tanh --n_latent 10 --name celeba_lbfgs --dataset celeba --dataroot /home/shared/datasets/celeba --latent_optimizer lbfgs --num_lbfgs_steps 20 --criterion vgg --img_size 64 --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 --nc 3  --display_freq 240 --lr 0.002 

##======CELEBA======

/home/arash/anaconda3/bin/python -u /home/arash/Desktop/SOG/train.py --gpu_ids 2,3,4,5 --n_deconv 5 --last_activation tanh --n_latent 8 --n_rounds 3 --name celeba --dataset celeba --dataroot /home/shared/datasets/celeba --img_size 64 --batch_size 48 --niter 5 --niter_decay 5 --save_epoch_freq 1 --nc 3  --display_freq 240 --lr 0.002
