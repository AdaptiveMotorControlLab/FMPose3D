#Train CFM
# python3 main_CFM.py --train --model model_GUMLP --layers 4 --lr 1e-3 --lr_decay 0.98 --nepoch 100 --sample_steps 3 --gpu 0 --debug
#Test CFM
python3 main_CFM.py --reload --previous_dir "./debug/250906_2235_46" --model model_GUMLP --sample_steps 3 --test_augmentation True --layers 4 --gpu 0