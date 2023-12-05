import pickle
import numpy as np
import os

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_anomaly(name):
    res = pkl_load(name)
    return res['train_data'], res['train_labels'], \
           res['test_data'],  res['test_labels']

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_anomaly("./cleaned_dataset/ucr_data.pt")
    train_data = {}
    valid_data = {}

    # Initialize the folder to save the trained model for each dataset
    run_dir = f'dataset/'
    os.makedirs(run_dir, exist_ok=True)

    for id in train_x.keys():
        value = train_x[id]
        # Data standardization (x-u)/std
        mean, std = value.mean(), value.std()
        value = (value - mean) / std
        
        # Split 10% of the data as validate set
        train_value, valid_value = np.split(value,[int(0.9 * len(value))])
        train_data[id] = train_value
        valid_data[id] = valid_value

        # Update the test data by standardization 
        test_x[id] = (test_x[id]- mean) / std

    output_file = 'dataset/ucr_data.pt'
    with open(output_file, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'valid_data': valid_data,
                'test_data': test_x,
                'test_labels': test_y,
            }, f)