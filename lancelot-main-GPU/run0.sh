# for method in krum #fedavg krum trimmed_mean fang
# do
#     /home/syjiang/anaconda3/bin/python main.py --gpu 0 --method $method --tsboard --c_frac 0.0 --quantity_skew
# done

python main.py --gpu 0 --method krum --tsboard --c_frac 0.0 --quantity_skew --dataset ImageNet