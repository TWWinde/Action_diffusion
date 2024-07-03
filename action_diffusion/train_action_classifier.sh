python3 /scratch/users/tang/Action_diffusion/action_diffusion/action_classifier.py \
--root /scratch/users/tang/data/niv/preprocessed_data \
--num_thread_reader 8 \
--cudnn_benchmark 1 \
--pin_memory \
--checkpoint_dir whl \
--batch_size 256 \
--batch_size_val 256 \
--evaluate \
--dataset crosstask \
--resume \
--horizon 3 \
--class_dim 48 \
--observation_dim 768