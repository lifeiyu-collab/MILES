dataset='ImageNet'
algorithm=('MILES')
data_dir='/mlspace/datasets/ImageNet/'
max_epoch=10
net=ViT-B/16
task='img_dg'
net_modified="${net//\//-}"
#---------------------------------------------------------ImageNet ViT 16-----------------------------------------------------------------------
gpu_ids=(0)  # 多张 GPU 的 ID 列表
gpu_ids_str=$(IFS=,; echo "${gpu_ids[*]}")
for lr in 0.002; do
for seed in 3 4 5; do
for i in 0; do
  python ../train_imagenet.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output '../scripts/'$dataset'_'$net_modified/${algorithm[i]} \
 --dataset $dataset --algorithm ${algorithm[i]}  --seed $seed --gpu_id $gpu_ids_str --N_WORKERS 4 --lr $lr --schuse  --schusech 'cos' --alpha 2 --beta 1 --shots_per_class 16 --warmup_epoch 1 --T 10
done
done
done
