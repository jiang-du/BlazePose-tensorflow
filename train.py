#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
import time
from model import BlazePose
from config import total_epoch, train_mode, continue_train, show_batch_loss
from analysis import save_record, load_record

if train_mode:
    from data import finetune_train as train_dataset
    from data import finetune_validation as test_dataset
    loss_func = tf.keras.losses.MeanSquaredError()
else:
    from data import train_dataset, test_dataset
    loss_func = tf.keras.losses.BinaryCrossentropy()

model = BlazePose()

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(y_true=targets, y_pred=model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# continue train
if continue_train > 0:
    model.load_weights(checkpoint_path.format(epoch=continue_train))
    # continue recording
    train_loss_results, train_accuracy_results, val_accuracy_results = load_record()
else:
    if train_mode:
        # start fine-tune
        from config import best_pre_train
        model.load_weights(checkpoint_path.format(epoch=best_pre_train))
    
    # start from epoch 0
    # Initial for record of the training process
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []

if train_mode:
    # finetune
    for layer in model.layers[0:16]:
        print(layer)
        layer.trainable = False
else:
    # pre-train
    for layer in model.layers[16:24]:
        print(layer)
        layer.trainable = False

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end="  Start train.\n")
# validata initial loaded model
val_accuracy = tf.keras.metrics.MeanSquaredError()
for x, y in test_dataset:
    val_accuracy(y, model(x))
print("Initial Validation accuracy: {:.5%}".format(val_accuracy.result()))

# make sure continue has any epoch to train
assert(continue_train < total_epoch)

for epoch in range(continue_train, total_epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()
    val_accuracy = tf.keras.metrics.MeanSquaredError()

    # Training loop
    if show_batch_loss:
        batch_index = 0
    for x, y in train_dataset:
        # Optimize
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Add current batch loss
        epoch_loss_avg(loss_value)
        # Calculate error from Ground truth
        epoch_accuracy(y, model(x))
        
        if show_batch_loss:
            print("Epoch {:03d}, Batch {:03d}: Train Loss: {:.3f}".format(epoch,
                batch_index,
                loss_value
            ))
            batch_index += 1
    
    # Record loss and accuracy
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # Train loss at epoch
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Epoch {:03d}: Train Loss: {:.3f}, Accuracy: {:.5%}".format(
        epoch,
        epoch_loss_avg.result(),
        epoch_accuracy.result()
    ))
    
    if not((epoch + 1) % 5):
        # validata and save weight every 5 epochs
        for x, y in test_dataset:
            val_accuracy(y, model(x))
        print("Epoch {:03d}, Validation accuracy: {:.5%}".format(epoch, val_accuracy.result()))
        model.save_weights(checkpoint_path.format(epoch=epoch))
        val_accuracy_results.append(val_accuracy.result())

        # save the training record at every validation epoch
        save_record(train_loss_results, train_accuracy_results, val_accuracy_results)

model.summary()

print("Finish training.")
