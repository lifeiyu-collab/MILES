dataset='office-home'
algorithm=('MILES')
data_dir='/mlspace/datasets/OfficeHome/'
max_epoch=50
net=ViT-B/16
task='img_dg'
net_modified="${net//\//-}"
#---------------------------------------------------------Office-Home ResNet 50-----------------------------------------------------------------------
gpu_id=0
for lr in 0.002; do
for test_envs in 0 1 2 3; do
for seed in 5; do
for i in 0; do
  python ../train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output '../scripts/'$dataset'_'$net_modified/'env'$test_envs/${algorithm[i]} \
  --test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]}  --seed $seed --gpu_id $gpu_id --N_WORKERS 4 --lr $lr --schuse  --schusech 'cos' --alpha 2 --beta 1 --optimizer 'SGD'  --T 10
done
done
done
done