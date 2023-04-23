import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import tarfile
import tempfile
import tensorflow as tf
import urllib.request 


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_CIFAR_labels():
    batch = unpickle('data/cifar-10-batches-py/batches.meta')
    class_names = [i.decode('utf-8') for i in batch[b'label_names']]
    class_labels = np.arange(0, 10, 1)
    class_dict = dict(zip(class_labels, class_names))
    return class_dict

def create_CIFAR_dataset(target_url, local_directory, split='train'):
    '''
    Purpose:
    --------
    Download and extract a tar.gz file from a 
    user-specified URL, and download to a local
    directory specified by the user
    
    Arguments:
    ----------
        target_url (string): A url containing a .tar.gz 
            extension to be used for extraction/download
        
        local_directory (string): A local file path to 
            extract the .tar.gz file into 
        
        split (boolean): Specifies whether the download 
            is for a training dataset (40k images), 
            validation dataset (10k images), or 
            for the test dataset (10k images)
            
    Returns:
    --------
    Returns a TensorFlow Dataset object of RGB images (32x32x3)
    and labels (0-9) based on the CIFAR-10 dataset
    '''
    # Create the local directory if it doesn't exist
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Download the dataset to a temporary file
    tmp_path = os.path.join(local_directory, 'tmp.tar.gz')
    urllib.request.urlretrieve(target_url, tmp_path)

    # Extract the dataset to local_directory/cifar-10-batches-py
    with tarfile.open(tmp_path, 'r:gz') as tar:
        tar.extractall(local_directory)

    # Remove the temporary file
    os.remove(tmp_path)

    # Load Batches for Training 
    test_batch = 'test_batch'
    
    if split == 'train':
        # Get list of data_batch filenames for CIFAR_10
        train_batches = [f'data_batch_{i}' for i in range(1, 5)]
        
        # Encode data into list structures
        image_data = []
        label_data = []

        for i in train_batches:
            batch = unpickle(f'data/cifar-10-batches-py/{i}')
            image_data.append(np.reshape(batch[b'data'], (10000, 32, 32, 3), order='F'))
            label_data.append(batch[b'labels'])
         
        # Merge all image and label data 
        images = np.concatenate(image_data)
        labels = np.concatenate(label_data)
        
        # Generate TF Dataset Object from the images/label data
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset

    if split == 'validation':
        # Get list of data_batch filenames for CIFAR_10
        train_batches = [f'data_batch_{i}' for i in range(5, 6)]
        
        # Encode data into list structures
        image_data = []
        label_data = []

        for i in train_batches:
            batch = unpickle(f'data/cifar-10-batches-py/{i}')
            image_data.append(np.reshape(batch[b'data'], (10000, 32, 32, 3), order='F'))
            label_data.append(batch[b'labels'])
         
        # Merge all image and label data 
        images = np.concatenate(image_data)
        labels = np.concatenate(label_data)
        
        # Generate TF Dataset Object from the images/label data
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset
    
    if split == 'test': 
        batch = unpickle(f'data/cifar-10-batches-py/test_batch')
        images = np.reshape(batch[b'data'], (10000, 32, 32, 3), order='F')
        labels = batch[b'labels']
        
        # Generate TF Dataset Object from the images/label data
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset
    
def visualize_dataset_counts(dataset, class_dict, dataset_name):
    labels_dataset = dataset.map(lambda image, label: label)
    labels = tf.concat([labels_batch for labels_batch in labels_dataset], axis=0).numpy()
    freq_counts = collections.Counter(labels)
    
    percentages = [(i/sum(freq_counts.values())) * 100 for i in freq_counts.values()]
    counts = freq_counts.values()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # plot the first dataset in the first subplot
    axs[0].pie(percentages, labels=class_dict.values(), autopct='%1.1f%%')
    axs[0].axis('equal')
    axs[0].set_title(f'Dataset: {dataset_name}\nDistribution of CIFAR-10 Classes\n')

    # plot the second dataset in the second subplot
    axs[1].bar(class_dict.values(), counts, width=0.5, align='edge')
    axs[1].set_xticks(np.arange(len(class_dict.values())) + 0.5/3, class_dict.values(), rotation=45)
    axs[1].set_title(f'Dataset: {dataset_name}\nCounts of CIFAR-10 Labels\n')

    # display the figure
    fig.tight_layout()
    plt.show()        
        
def proba_classification_report(predictions, truth, predict_probas, threshold, class_names):
    df = pd.DataFrame({
        'Predicted Class': predictions,
        'Truth Label': truth,
        'Prediction Probability': predict_probas
    })
    original_len = len(df)
    df = df.loc[df['Prediction Probability'] > threshold]
    new_len = len(df)
    print(f"Classification Report for {threshold}:\nPredictions Retained: {round(new_len/original_len, 2)*100}\n{classification_report(df['Truth Label'], df['Predicted Class'], target_names = class_names)}")
    
    disp = ConfusionMatrixDisplay.from_predictions(
        df['Truth Label'],
        df['Predicted Class'],
        display_labels=list(class_names),
        xticks_rotation=45 
    )

    print(f"Confusion Matrix:\n{disp}")
    