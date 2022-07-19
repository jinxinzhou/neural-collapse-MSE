################################### MSE ############################################

#------------------------------- different optimizer CIFAR10 ------------------------------

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_Adam_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --optimizer Adam --lr 0.001
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_Adam_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_LBFGS_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 512
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_LBFGS_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/

# ----------------------------------- ETF ----------------------------------------------------
python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/ --SOTA

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --ETF_fc
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true/ --SOTA --ETF_fc

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --fixdim 10
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true/ --SOTA --fixdim 10

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --ETF_fc --fixdim 10
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true/ --SOTA --ETF_fc --fixdim 10

# -------------------------------- different dimension ----------------------------------------
python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_5_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --fixdim 5
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_5_sota_true/ --SOTA --fixdim 5

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_8_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --fixdim 8
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_8_sota_true/ --SOTA --fixdim 8

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_9_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --fixdim 9
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_9_sota_true/ --SOTA --fixdim 9

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_512_sota_true --dataset cifar10 --loss MSE --optimizer SGD --lr 0.05 --SOTA --fixdim 512
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_512_sota_true/ --SOTA --fixdim 512

# -------------------------------------- rescale MSE -------------------------------------
# effect of alpha(which is notation in paper, but k is used in code)
python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_1 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 1 --k 1
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_1/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_5 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 1 --k 5
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_5/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_15 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 1 --k 15
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_15/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_20 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 1 --k 20
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_20/ --SOTA
# effect of M
python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_3_k_1 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 3 --k 1
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_3_k_1/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_5_k_1 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 5 --k 1
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_5_k_1/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_7_k_1 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 7 --k 1
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_7_k_1/ --SOTA

python train_1st_order.py --gpu_id 0 --uid mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_10_k_1 --dataset mini_imagenet --loss RescaledMSE --optimizer SGD --lr 0.01 --SOTA --M 10 --k 1
python validate_NC.py --gpu_id 0 --dataset mini_imagenet --batch_size 1024 --load_path model_weights/mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_10_k_1/ --SOTA