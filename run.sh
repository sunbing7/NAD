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

