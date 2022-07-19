################################### CE ############################################

#------------------------------- different optimizer CIFAR10 ------------------------------

#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer Adam --lr 0.001
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_Adam_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 512
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_LBFGS_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/

# ----------------------------------- ETF ----------------------------------------------------
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/ --SOTA
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --ETF_fc
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true/ --SOTA --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --fixdim 10
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true/ --SOTA --fixdim 10
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --ETF_fc --fixdim 10
#python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true/ --SOTA --ETF_fc --fixdim 10

# -------------------------------- different dimension ----------------------------------------
python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_5_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --fixdim 5
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_5_sota_true/ --SOTA --fixdim 5

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_8_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --fixdim 8
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_8_sota_true/ --SOTA --fixdim 8

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_9_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --fixdim 9
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_9_sota_true/ --SOTA --fixdim 9

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_512_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA --fixdim 512
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_512_sota_true/ --SOTA --fixdim 512