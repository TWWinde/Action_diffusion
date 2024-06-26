data="crosstask"
horizon=3
attn="WithAttention"
mask_type="multi_add"
if [ $data = "crosstask" ]
then
	action_dim=105
	class_dim=18
	act_emb="dataset/crosstask/act_lang_emb.pkl"
	diffusion_step=200
	train_step=200
	json_train="dataset/crosstask/crosstask_release/train_split_T$horizon.json"
	json_val="dataset/crosstask/crosstask_release/crosstask_mlp_T$horizon.json"
	epoch=120
	lr=5e-4
fi
if [ $data = "coin" ]
then
	action_dim=778
	class_dim=180
	act_emb="dataset/coin/steps_info.pickle"
	diffusion_step=200
	train_step=200
	json_train="dataset/coin/train_split_T$horizon.json"
	json_val="dataset/coin/coin_mlp_T$horizon.json"
	epoch=800
	lr=1e-5
fi
if [ $data = "NIV" ]
then
	action_dim=48
	class_dim=5
	act_emb="dataset/NIV/niv_act_embeddings.pickle"
	diffusion_step=50
	train_step=50
	json_train="dataset/NIV/train_split_T$horizon.json"
	json_val="dataset/NIV/NIV_mlp_T$horizon.json"
	epoch=130
	if [ $horizon -eq 3 ]
	then
		lr=1e-4
	fi
	if [ $horizon -eq 4 ]
	then
		lr=3e-4
	fi
fi
python3 main_distributed_act.py \
--multiprocessing-distributed \
--num_thread_reader=8 \
--cudnn_benchmark=1 \
--pin_memory \
--checkpoint_dir=whl \
--resume \
--batch_size=256 \
--batch_size_val=256 \
--evaluate \
--dataset ${data} \
--horizon ${horizon} \
--mask_type ${mask_type} \
--attn ${attn} \
--act_emb_path ${act_emb} \
--action_dim ${action_dim} \
--class_dim ${class_dim} \
--n_diffusion_steps ${diffusion_step} \
--n_train_steps ${train_step} \
--json_path_train ${json_train} \
--json_path_val ${json_val} \
--epochs ${epoch} \
--lr ${lr} \
--use_cls_mask True
