python3 temp.py \
--multiprocessing-distributed \
--num_thread_reader 8 \
--cudnn_benchmark 1 \
--pin_memory \
--checkpoint_dir whl \
--batch_size 32 \
--batch_size_val 32 \
--evaluate \
--dataset crosstask \
--resume \
--horizon 4 \
--action_dim 105 \
--class_dim 18 \
--observation_dim 1536 \
--json_path_val dataset/crosstask/crosstask_release/test_split_T4.json \
--json_path_train dataset/crosstask/crosstask_release/train_split_T4.json \
--checkpoint_mlp epoch0010_act_T4.pth.tar
