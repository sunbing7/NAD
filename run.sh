#######################################################################################################################################
#cifar resnet18
#test
python train_semantic.py --mode=test --t_attack=green --target_label=6 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --epochs=10 --batch_size=64 --data_name=CIFAR10 --s_name=resnet18 --in_model=_cifar10_green.pt
python train_semantic.py --mode=test --t_attack=sbg --target_label=9 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --epochs=10 --batch_size=64 --data_name=CIFAR10 --s_name=resnet18 --in_model=_cifar10_sbg.pt

#finetune
python train_semantic.py --lr=0.0001 --mode=finetune --t_attack=green --target_label=6 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --s_name=resnet18 --in_model=_cifar10_green.pt --out_model=_cifar10_green_t.pt
python train_semantic.py --lr=0.0001 --mode=finetune --t_attack=sbg --target_label=9 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --s_name=resnet18 --in_model=_cifar10_sbg.pt --out_model=_cifar10_sbg_t.pt

#mitigation
python train_semantic.py --lr=0.0001 --mode=nad --t_attack=green --target_label=6 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/resnet18_cifar10_green.pt --t_model=./weight/erasing_net/resnet18_cifar10_green_t.pt.tar --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --t_name=resnet18 --s_name=resnet18 --out_model=_cifar10_green_fixed.pth
python train_semantic.py --lr=0.0001 --mode=nad --t_attack=sbg --target_label=9 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/resnet18_cifar10_sbg.pt --t_model=./weight/erasing_net/resnet18_cifar10_sbg_t.pt.tar --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --t_name=resnet18 --s_name=resnet18 --out_model=_cifar10_sbg_fixed.pth

#cifar resnet50
#test
python train_semantic.py --mode=test --t_attack=green --target_label=6 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --epochs=10 --batch_size=64 --data_name=CIFAR10 --s_name=resnet50 --in_model=_cifar10_green.pt
python train_semantic.py --mode=test --t_attack=sbg --target_label=9 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --epochs=10 --batch_size=64 --data_name=CIFAR10 --s_name=resnet50 --in_model=_cifar10_sbg.pt

#finetune
python train_semantic.py --lr=0.0001 --mode=finetune --t_attack=green --target_label=6 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --s_name=resnet50 --in_model=_cifar10_green.pt --out_model=_cifar10_green_t.pt
python train_semantic.py --lr=0.0001 --mode=finetune --t_attack=sbg --target_label=9 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --s_name=resnet50 --in_model=_cifar10_sbg.pt --out_model=_cifar10_sbg_t.pt

#mitigation
python train_semantic.py --lr=0.0001 --mode=nad --t_attack=green --target_label=6 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/resnet50_cifar10_green.pt --t_model=./weight/erasing_net/resnet50_cifar10_green_t.pt.tar --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --t_name=resnet50 --s_name=resnet50 --out_model=_cifar10_green_fixed.pth
python train_semantic.py --lr=0.0001 --mode=nad --t_attack=sbg --target_label=9 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/resnet50_cifar10_sbg.pt --t_model=./weight/erasing_net/resnet50_cifar10_sbg_t.pt.tar --data_path=./data/CIFAR10/cifar_dataset.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=CIFAR10 --t_name=resnet50 --s_name=resnet50 --out_model=_cifar10_sbg_fixed.pth


#gtsrb vgg11_bn
#test
python train_semantic.py --mode=test --t_attack=dtl --num_class=43 --target_label=0 --checkpoint_root=./weight/erasing_net --data_path=./data/GTSRB/gtsrb.h5 --epochs=10 --batch_size=64 --data_name=GTSRB --s_name=vgg11_bn --in_model=_gtsrb_dtl.pt
python train_semantic.py --mode=test --t_attack=dkl --num_class=43 --target_label=6 --checkpoint_root=./weight/erasing_net --data_path=./data/GTSRB/gtsrb.h5 --epochs=10 --batch_size=64 --data_name=GTSRB --s_name=vgg11_bn --in_model=_gtsrb_dkl.pt

#finetune
python train_semantic.py --lr=0.0001 --mode=finetune --num_class=43 --t_attack=dtl --target_label=0 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/GTSRB/gtsrb.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=GTSRB --t_name=vgg11_bn --s_name=vgg11_bn --in_model=_gtsrb_dtl.pt --out_model=_gtsrb_dtl_t.pt
python train_semantic.py --lr=0.0001 --mode=finetune --num_class=43 --t_attack=dkl --target_label=6 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/GTSRB/gtsrb.h5 --batch_size=64 --beta1=0 --beta2=0 --beta3=0 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=GTSRB --t_name=vgg11_bn --s_name=vgg11_bn --in_model=_gtsrb_dkl.pt --out_model=_gtsrb_dkl_t.pt

#mitigation
python train_semantic.py --lr=0.0001 --mode=nad --num_class=43 --t_attack=dtl --target_label=0 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/vgg11_bn_gtsrb_dtl.pt --t_model=./weight/erasing_net/vgg11_bn_gtsrb_dtl_t.pt.tar --data_path=./data/GTSRB/gtsrb.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=GTSRB --t_name=vgg11_bn --s_name=vgg11_bn --out_model=_gtsrb_dtl_fixed.pt
python train_semantic.py --lr=0.0001 --mode=nad --num_class=43 --t_attack=dkl --target_label=6 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/vgg11_bn_gtsrb_dkl.pt --t_model=./weight/erasing_net/vgg11_bn_gtsrb_dkl_t.pt.tar --data_path=./data/GTSRB/gtsrb.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=GTSRB --t_name=vgg11_bn --s_name=vgg11_bn --out_model=_gtsrb_dkl_fixed.pt

#fmnsit mobilenetv2
#test
python train_semantic.py --mode=test --t_attack=stripet --num_class=10 --target_label=2 --checkpoint_root=./weight/erasing_net --data_path=./data/FMNIST/fmnist.h5 --epochs=10 --batch_size=64 --data_name=FMNIST --s_name=MobileNetV2 --in_model=_fmnist_stripet.pt
python train_semantic.py --mode=test --t_attack=plaids --num_class=10 --target_label=4 --checkpoint_root=./weight/erasing_net --data_path=./data/FMNIST/fmnist.h5 --epochs=10 --batch_size=64 --data_name=FMNIST --s_name=MobileNetV2 --in_model=_fmnist_plaids.pt

#finetune
python train_semantic.py --lr=0.001 --mode=finetune --num_class=10 --t_attack=stripet --target_label=2 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/FMNIST/fmnist.h5 --batch_size=64 --data_name=FMNIST --s_name=MobileNetV2 --in_model=_fmnist_stripet.pt --out_model=_fmnist_stripet_t.pt
python train_semantic.py --lr=0.001 --mode=finetune --num_class=10 --t_attack=plaids --target_label=4 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/FMNIST/fmnist.h5 --batch_size=64 --data_name=FMNIST --s_name=MobileNetV2 --in_model=_fmnist_plaids.pt --out_model=_fmnist_plaids_t.pt

#mitigation
python train_semantic.py --lr=0.001 --mode=nad --num_class=10 --t_attack=stripet --target_label=2 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/MobileNetV2_fmnist_stripet.pt --t_model=./weight/erasing_net/MobileNetV2_fmnist_stripet_t.pt.tar --data_path=./data/FMNIST/fmnist.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=FMNIST --t_name=MobileNetV2 --s_name=MobileNetV2 --out_model=_fmnist_stripet_fixed.pt
python train_semantic.py --lr=0.001 --mode=nad --num_class=10 --t_attack=plaids --target_label=4 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/MobileNetV2_fmnist_plaids.pt --t_model=./weight/erasing_net/MobileNetV2_fmnist_plaids_t.pt.tar --data_path=./data/FMNIST/fmnist.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=FMNIST --t_name=MobileNetV2 --s_name=MobileNetV2 --out_model=_fmnist_plaids_fixed.pt


#mnsitm densenet
#test
python train_semantic.py --mode=test --t_attack=blue --num_class=10 --target_label=3 --checkpoint_root=./weight/erasing_net --data_path=./data/mnist_m/mnistm.h5 --epochs=10 --batch_size=64 --data_name=mnistm --s_name=densenet --in_model=_mnistm_blue.pt
python train_semantic.py --mode=test --t_attack=black --num_class=10 --target_label=3 --checkpoint_root=./weight/erasing_net --data_path=./data/mnist_m/mnistm.h5 --epochs=10 --batch_size=64 --data_name=mnistm --s_name=densenet --in_model=_mnistm_black.pt

#finetune
python train_semantic.py --lr=0.001 --mode=finetune --num_class=10 --t_attack=blue --target_label=3 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/mnist_m/mnistm.h5 --batch_size=64 --data_name=mnistm --s_name=densenet --in_model=_mnistm_blue.pt --out_model=_mnistm_blue_t.pt
python train_semantic.py --lr=0.001 --mode=finetune --num_class=10 --t_attack=black --target_label=3 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/mnist_m/mnistm.h5 --batch_size=64 --data_name=mnistm --s_name=densenet --in_model=_mnistm_black.pt --out_model=_mnistm_black_t.pt

#mitigation
python train_semantic.py --lr=0.001 --mode=nad --num_class=10 --t_attack=blue --target_label=3 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/densenet_mnistm_blue.pt --t_model=./weight/erasing_net/densenet_mnistm_blue_t.pt.tar --data_path=./data/mnist_m/mnistm.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=mnistm --t_name=densenet --s_name=densenet --out_model=_mnistm_blue_fixed.pt
python train_semantic.py --lr=0.001 --mode=nad --num_class=10 --t_attack=black --target_label=3 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/densenet_mnistm_black.pt --t_model=./weight/erasing_net/densenet_mnistm_black_t.pt.tar --data_path=./data/mnist_m/mnistm.h5 --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=mnistm --t_name=densenet --s_name=densenet --out_model=_mnistm_black_fixed.pt


#asl MobileNet
#test
python train_semantic.py --mode=test --t_attack=A --num_class=29 --target_label=4 --checkpoint_root=./weight/erasing_net --data_path=./data/asl/attack_A/ --epochs=10 --batch_size=64 --data_name=asl --s_name=MobileNet --in_model=_asl_A.pt
python train_semantic.py --mode=test --t_attack=Z --num_class=29 --target_label=11 --checkpoint_root=./weight/erasing_net --data_path=./data/asl/attack_Z/ --epochs=10 --batch_size=64 --data_name=asl --s_name=MobileNet --in_model=_asl_Z.pt

#finetune
python train_semantic.py --lr=0.001 --mode=finetune --num_class=29 --t_attack=A --target_label=4 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/asl/attack_A/  --batch_size=64 --data_name=asl --s_name=MobileNet --in_model=_asl_A.pt --out_model=_asl_A_t.pt
python train_semantic.py --lr=0.001 --mode=finetune --num_class=29 --t_attack=Z --target_label=11 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/asl/attack_Z/ --batch_size=64 --data_name=asl --s_name=MobileNet --in_model=_asl_Z.pt --out_model=_asl_Z_t.pt

#mitigation
python train_semantic.py --lr=0.001 --mode=nad --num_class=29 --t_attack=A --target_label=4 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/MobileNet_asl_A.pt --t_model=./weight/erasing_net/MobileNet_asl_A_t.pt.tar --data_path=./data/asl/attack_A/ --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=asl --t_name=MobileNet --s_name=MobileNet --out_model=_asl_A_fixed.pt
python train_semantic.py --lr=0.001 --mode=nad --num_class=29 --t_attack=Z --target_label=11 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/MobileNet_asl_Z.pt --t_model=./weight/erasing_net/MobileNet_asl_Z_t.pt.tar --data_path=./data/asl/attack_Z/ --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=asl --t_name=MobileNet --s_name=MobileNet --out_model=_asl_Z_fixed.pt


#caltech shufflenetv2
#test
python train_semantic.py --mode=test --t_attack=brain --num_class=101 --target_label=42 --checkpoint_root=./weight/erasing_net --data_path=./data/caltech/bl_brain/ --epochs=10 --batch_size=64 --data_name=caltech --s_name=shufflenetv2 --in_model=_caltech_brain.pt
python train_semantic.py --mode=test --t_attack=g_kan --num_class=101 --target_label=1 --checkpoint_root=./weight/erasing_net --data_path=./data/caltech/g_kan/ --epochs=10 --batch_size=64 --data_name=caltech --s_name=shufflenetv2 --in_model=_caltech_g_kan.pt

#finetune
python train_semantic.py --lr=0.001 --mode=finetune --num_class=101 --t_attack=brain --target_label=42 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/caltech/bl_brain/ --batch_size=64 --data_name=caltech --s_name=shufflenetv2 --in_model=_caltech_brain.pt --out_model=_caltech_brain_t.pt
python train_semantic.py --lr=0.001 --mode=finetune --num_class=101 --t_attack=g_kan --target_label=1 --epochs=10 --checkpoint_root=./weight/erasing_net --data_path=./data/caltech/g_kan/ --batch_size=64 --data_name=caltech --s_name=shufflenetv2 --in_model=_caltech_g_kan.pt --out_model=_caltech_g_kan_t.pt

#mitigation
python train_semantic.py --lr=0.001 --mode=nad --num_class=101 --t_attack=brain --target_label=42 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/shufflenetv2_caltech_brain.pt --t_model=./weight/erasing_net/shufflenetv2_caltech_brain_t.pt.tar --data_path=./data/caltech/bl_brain/ --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=caltech --t_name=shufflenetv2 --s_name=shufflenetv2 --out_model=_caltech_brain_fixed.pt
python train_semantic.py --lr=0.001 --mode=nad --num_class=101 --t_attack=g_kan --target_label=1 --epochs=10 --beta1=500 --beta2=1000 --beta3=1000 --checkpoint_root=./weight/erasing_net --s_model=./weight/erasing_net/shufflenetv2_caltech_g_kan.pt --t_model=./weight/erasing_net/shufflenetv2_caltech_g_kan_t.pt.tar --data_path=./data/caltech/g_kan/ --batch_size=64 --threshold_clean=70 --threshold_bad=0 --save=1 --data_name=caltech --t_name=shufflenetv2 --s_name=shufflenetv2 --out_model=_caltech_g_kan_fixed.pt
