# save the training record
import json
import numpy as np
from config import json_name

def save_record(train_loss_results, train_accuracy_results, val_accuracy_results):
    train_record = dict()
    train_record["train_loss"] = list(np.float64(train_loss_results))
    train_record["train_accuracy"] = list(np.float64(train_accuracy_results))
    train_record["val_accuracy"] = list(np.float64(val_accuracy_results))
    with open(json_name, 'w') as f:
        json.dump(train_record, f)
    return 0

def load_record():
    with open(json_name, 'r') as f:
        train_record = json.load(f)
    # convert list to numpy
    train_loss_results = np.float64(train_record["train_loss"])
    train_accuracy_results = np.float64(train_record["train_accuracy"])
    val_accuracy_results = np.float64(train_record["val_accuracy"])
    return train_loss_results, train_accuracy_results, val_accuracy_results
