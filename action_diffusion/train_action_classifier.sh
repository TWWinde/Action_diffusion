python3 /scratch/users/tang/Action_diffusion/action_diffusion/action_classifier.py \
--root /scratch/users/tang/data/niv/train_16_onlystart_pooled_aug_x6 \
--num_thread_reader 8 \
--cudnn_benchmark 1 \
--pin_memory \
--checkpoint_dir whl \
--batch_size 256 \
--batch_size_val 256 \
--evaluate \
--resume \
--horizon 3 \
--class_dim 48 \
--observation_dim 768 \
--dataset niv


#python3 /scratch/users/tang/Action_diffusion/action_diffusion/action_classifier.py \
#--root /scratch/users/tang/data/COIN/* \
#--num_thread_reader 8 \
#--cudnn_benchmark 1 \
#--pin_memory \
#--checkpoint_dir whl \
#--batch_size 256 \
#--batch_size_val 256 \
#--evaluate \
#--resume \
#--horizon 3 \
#--class_dim 778 \
#--observation_dim 768 \
#--dataset coin


#python3 /scratch/users/tang/Action_diffusion/action_diffusion/action_classifier.py \
#--root /scratch/users/tang/data/crosstask_release/* \
#--num_thread_reader 8 \
#--cudnn_benchmark 1 \
#--pin_memory \
#--checkpoint_dir whl \
#--batch_size 256 \
#--batch_size_val 256 \
#--evaluate \
#--resume \
#--horizon 3 \
#--class_dim  \
#--observation_dim 768 \
#--dataset crosstask
