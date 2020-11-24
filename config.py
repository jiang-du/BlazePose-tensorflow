num_joints = 14     # lsp dataset

batch_size = 256    # 256 is best for RTX 3090 (per GPU)
total_epoch = 500
gpu_dynamic_memory = 0
gaussian_sigma = 4

# Train mode: 0-pre-train, 1-finetune
train_mode = 0

show_batch_loss = 0
continue_train = 0  # 0 for random initialize, >0 for num epoch

if train_mode:
    best_pre_train = 434 # num of epoch where the training loss drops but testing accuracy achieve the optimal

json_name = "train_record.json" if train_mode else "train_record_pretrain.json"
