# suboptimal encapsulation
# relies on the following global variables to be set
# IMG_SIZE
# RESULT_CSV
# BATCH_SIZE
# class_names

IMG_SIZE = 224
BATCH_SIZE = 32
RESULT_CSV = './results.csv'    # results_200_200.csv lists the result from runs with images of size 200x200


import csv
import os
from datetime import datetime

import re

from skimage import color, exposure, transform, io
from PIL import Image

import numpy as np
import pandas as pd

from itertools import chain

# graphics
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf

from keras.callbacks import EarlyStopping


# Sample dataframes to size
def sample_to_n(df,n):    
    """ sampling dataframe to desired size (downsample, upsample handled dynamically)
    """
    # Group by class
    dfs = []
    
    if type(n) == int:
        n_count = n
    elif n == 'max':
        n_count = int(df['category'].value_counts().max())
    elif n == 'med':
        n_count = int(df['category'].value_counts().median())
    elif n == 'mean':
        n_count = int(df['category'].value_counts().mean())
    elif n == 'min':
        n_count = int(df['category'].value_counts().min())
    else:
        raise ValueError(f"Unsupported sampling size: {n}")

    for label, group in df.groupby('category'):
        replace = len(group) < n_count  # Only upsample if needed
        upsampled = resample(group, 
                            replace=replace, 
                            n_samples=n_count, 
                            random_state=42)
        dfs.append(upsampled)

    return pd.concat(dfs)



# scaling images to target size (as squares!)
#
def load_single_img_v2(path, preprocess_fn=None):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    if preprocess_fn:
        image = preprocess_fn(image)
    else:
        image = image / 255.0

    return tf.expand_dims(image, axis=0)  # Add batch dimension


# image augmentation for training
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    
    # Small random crop and resize to simulate zoom/shift
    crop_frac = tf.random.uniform([], 0.85, 1.0)
    crop_size = tf.cast(crop_frac * IMG_SIZE, tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    return image


# image loading and preprocessing
def load_and_preprocess(path, label, augment=False,preprocess_fn=None):
    """ Load and preprocess an image from a given path
    
    - augment: set to True to enable image augmentation
    
    return:
    - processed **image**, **label**  
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    if augment:
        image = augment_image(image)
    
    image = tf.cast(image, tf.float32)
    if preprocess_fn:
        image = preprocess_fn(image)
    else:
        image = image / 255
    return image, label


# create tf.Dataset
def mk_tf_dataset(df, shuffle=True, augment=False,preprocess_fn=None):
    """ Create a tensorflow dataset from a dataframe
    
    Arguments:
    - df: pandas dataframe
    - shuffle: enable shuffling
    - augment: augment the dataset
    """
    paths = df['relpath'].values
    labels = df['label'].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    ds = ds.map(lambda x, y: load_and_preprocess(x, y, augment=augment,preprocess_fn=preprocess_fn), 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# print evaluation and append results to results 
def print_evaluation(model,test_ds,test_df,acc_train=None,acc_val=None,model_id=None,lr=None,epochs=None,batchsize=None,csvout=RESULT_CSV,printout=True):
    """ Print model evaluation
    
    Arguments:
    - **model**: the model for evaluation
    - **test_ds** a tensorflow dataset for the test data
    - **test_df** a pandas dataframe for the test data (expects the true labels in 'label')
    - **acc_train** can be used to supply the final training accuracy from the fitting history for logging (otherwise, None is supplied)
    - **acc_val** can be used to supply the final validation accuracy from the fitting history for logging (otherwise, None is supplied)
    - **model_id**: a string identifier for the model for logging
    - **csvout**: filepath for csv file logging the results, set to '' to disable logging
    - **printout**: set to False if no printed output is desired
     
    return:
    - classification report for storage and further comparison
    """
    # Predict class probabilities
    y_pred_probs = model.predict(test_ds)

    # Convert probabilities to class predictions
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = test_df['label']  # original integer labels

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    class_names = (
    test_df[['label', 'category']]
    .drop_duplicates()
    .sort_values('label')['category']
    .tolist()
    )

    report = classification_report(y_true, y_pred, target_names=class_names)

    if printout:
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(report)
    
    
        # Predict & confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    if csvout:
        write_results(model_id,lr,epochs,batchsize,acc_train,acc_val,acc,prec,rec,f1,file=csvout)
    
    return report


# function to append results to csv
def write_results(modid,lr,epochs,batchsize,acc_train,acc_val,accuracy,precision,recall,f1,file=RESULT_CSV):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fields=[timestamp, modid, lr, epochs, batchsize, acc_train,acc_val, accuracy, precision, recall, f1]
    headers = ['time','model_id', 'learning_rate', 'epochs', 'batchsize', 'acc_train', 'acc_val', 'accuracy', 'precision', 'recall', 'f1_score']
    
    
    file_exists = os.path.exists(file)

    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(fields)
        
        
# to more easily compare
def print_side_by_side(*tables, padding=4):
    # Split each table into lines
    split_tables = [t.splitlines() for t in tables]
    
    # Find the max number of lines
    max_lines = max(len(t) for t in split_tables)
    
    # Pad shorter tables with empty lines
    split_tables = [
        t + [''] * (max_lines - len(t)) for t in split_tables
    ]
    
    # Join each line side-by-side
    for lines in zip(*split_tables):
        print((' ' * padding).join(line.ljust(30) for line in lines))

# train and evaluate model
def train_and_report(model,ds_train,ds_val,ds_test,test_df,logname,learnrate,epochs,batchsize,early_patience=5,returnhist=False):
    """Train and evaluate a model. 

    Args:
        model _Model): the model to train and evaluate
        ds_train (tensorflow.Dataset): Dataset for training
        ds_val (tensorflow.Dataset): Dataset for validation
        ds_test (tensorflow.Dataset): Dataset for testing
        test_df (pandas.DataFrame): DataFrame for the test set, needed for the labels
        logname (str): name of the model for logging
        learnrate (float): learning rate for the model (for logging only, set at model compilation)
        epochs (int): number of epochs to run
        early_patience (int, optional): Set the patience for EarlyStopping. Defaults to 5. Set to None to disable EarlyStopping

    Returns:
        str: returns the classification report for the test set
    """
    if early_patience:
        early_stop = EarlyStopping(monitor='val_loss', patience=early_patience, restore_best_weights=True)

        # Train model
        model_hist = model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_val,
            verbose=1,
            callbacks=early_stop
        )
    else:
        # Train model
        model_hist = model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_val,
            verbose=1
        )

    train_acc = model_hist.history['accuracy'][-1]
    val_acc = model_hist.history['val_accuracy'][-1]
    
    if returnhist:
        return model_hist, print_evaluation(model,
                            ds_test,
                            test_df,
                            train_acc,
                            val_acc,
                            logname,
                            learnrate,
                            epochs,
                            batchsize)
    else:
        return print_evaluation(model,
                            ds_test,
                            test_df,
                            train_acc,
                            val_acc,
                            logname,
                            learnrate,
                            epochs,
                            batchsize)
        

def plothist(md_hist,figsize=(12,5), export=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Accuracy
    axs[0].plot(md_hist.history['accuracy'])
    axs[0].plot(md_hist.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='lower right')

    # Loss
    axs[1].plot(md_hist.history['loss'])
    axs[1].plot(md_hist.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()

    if export:
        plt.savefig(export)
    plt.show()
    
    
def single_prediction(model,pic):
    y_pred_probs = model.predict(pic)

    # Convert probabilities to class predictions
    return y_pred_probs.argmax(axis=1)


def m2tex(model, modelName):
    stringlist = []
    model.summary(line_length=70, print_fn=lambda x: stringlist.append(x))

    # Filter out separator lines (rows with only '-' characters)
    stringlist = [line for line in stringlist if not set(line.strip()).issubset({'-', ' '})]

    stringlist = list(chain.from_iterable(line.splitlines() for line in stringlist))

    # Format layer rows
    for i in range(1, len(stringlist) - 4):  # skip header and footer
        line = stringlist[i]
        stringlist[i] = f"{line[0:31]} & {line[31:59]} & {line[59:]} \\\\"

    # Update header and footer
    stringlist[0] = f"Model: {modelName} \\\\"
    stringlist[1] += r" \\ \hline"      # Column headers
    stringlist[-4] += r" \\ \hline"     # Total params
    stringlist[-3] += r" \\"            # Trainable params
    stringlist[-2] += r" \\"            # Non-trainable params
    stringlist[-1] += r" \\ \hline"     # Final blank

    # Build LaTeX table
    prefix = [
        r"\begin{table}[]",
        r"\centering",
        r"\begin{tabular}{lll}",
        r"\hline"
    ]
    suffix = [
        r"\end{tabular}",
        rf"\caption{{Model summary for {modelName}.}}",
        rf"\label{{tab:model-summary}}",
        r"\end{table}"
    ]

    # Combine everything and escape LaTeX-sensitive characters
    full = prefix + stringlist + suffix
    out_str = "\n".join(full)
    out_str = out_str.replace("_", r"\_").replace("#", r"\#")

    print(out_str)
    

def m2tex_improved(model, modelName="MyModel"):
    output = []
    model.summary(print_fn=lambda x: output.append(x))
    
    # Split into actual lines
    lines = "\n".join(output).splitlines()

    in_table = False
    layer_lines = []
    summary_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove Unicode/ASCII table characters
        line = re.sub(r"[│┆┃╎┃║]+", "", line)
        line = re.sub(r"[─━═]+", "", line)

        if "Layer (type)" in line and "Output Shape" in line:
            in_table = True
            continue
        if in_table:
            if any(kw in line for kw in ["Total params", "Trainable params", "Non-trainable params"]):
                summary_lines.append(line)
                continue
            if re.match(r"=+|-+", line):
                continue
            parts = re.split(r"\s{2,}", line)
            if len(parts) == 3:
                # Extract just the layer type from: "name (Type)"
                match = re.search(r"\(([^)]+)\)", parts[0])
                layer_type = match.group(1) if match else parts[0]
                layer_lines.append([layer_type, parts[1], parts[2]])

    # LaTeX table output
    latex_lines = [
        r"\begin{table}[]",
        r"\centering",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Layer Type & Output Shape & Param \# \\ \midrule"
    ]

    for row in layer_lines:
        row = [col.replace("_", r"\_").replace("#", r"\#") for col in row]
        latex_lines.append(f"{row[0]} & {row[1]} & {row[2]} \\\\")

    latex_lines.append(r"\midrule")

    for line in summary_lines:
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip().replace("_", r"\_").replace("#", r"\#")
            val = parts[1].strip()
            latex_lines.append(rf"{key} & {val} \\")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Model summary for {modelName}}}",
        rf"\label{{tab:{modelName.lower().replace(' ', '-')}}}",
        r"\end{table}"
    ]

    print("\n".join(latex_lines))