num_joints = 14     # lsp
batch_size = 256    # 256 is best for RTX 3090 (per GPU)
total_epoch = 500
gpu_dynamic_memory = 0
gaussian_sigma = 4
train_mode = 0      # 0-pre-train, 1-finetune
show_batch_loss = 0
continue_train = 0  # 0 for random initialize, >0 for num epoch

json_name = "train_record.json" if train_mode else "train_record_pretrain.json"
