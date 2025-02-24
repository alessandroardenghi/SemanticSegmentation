import rasterio as rio
from rasterio.transform import xy

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import json


def read_tiff(datapoint_path):              # Function taken from notebook provided in assignment description
    img = rio.open(datapoint_path)
    img_array = img.read()
    nRows = img_array.shape[1]
    ncols = img_array.shape[2]
    Bands = img_array.shape[0]

    img_array = img_array.reshape(Bands, nRows*ncols).T
    img_array = img_array.reshape( nRows,ncols, Bands)
    
    return img_array


def plot_tiff(datapoint_path, figsize=(10, 10)):
    
    img_array = read_tiff(datapoint_path)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].imshow(img_array[:, :, :3]/255.0)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')  
    
    axes[0, 1].imshow(img_array[:, :, 3], 'gray')
    axes[0, 1].set_title('Infrared Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_array[:, :, 4], 'gray')
    axes[1, 0].set_title('Elevation Image')
    axes[1, 0].axis('off')
    
    word_labels = {
    0.0: 'Impervious Surface',
    1.0: 'Building',
    2.0: 'Tree',
    3.0: 'Low Vegetation',
    4.0: 'Car',
    5.0: 'Clutter/Background'
    }

    boundaries = np.arange(-0.5, 6.5, 1)
    cmap = plt.cm.viridis  
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    im = axes[1, 1].imshow(img_array[:, :, 5], cmap=cmap, norm=norm)
    axes[1, 1].set_title('Masks')

    cbar = plt.colorbar(im, ticks=np.arange(0, 6))
    cbar.ax.set_yticklabels([word_labels[i] for i in range(6)])
    plt.tight_layout()  
    plt.show()
    

def _bytes_feature(value):                                  # Function taken from Tensorflow Documentation
    
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):                                  # Function taken from Tensorflow Documentation
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_datapoint(datapoint_path):
    
    # Reading the tiff file
    img_array = read_tiff(datapoint_path)
    
    # Computing latitude and longitude as shown in the notebook on Canvas
    with rio.open(datapoint_path) as dataset:
        width, height = dataset.width, dataset.height
        center_x = width // 2
        center_y = height // 2
        lon, lat = xy(dataset.transform, center_y, center_x)
    
    
    # Splitting the image into RGB, IR, Elevation, and Mask channels.
    image_rgb = img_array[:, :, :3]                     # shape (224, 224, 3)
    ir = img_array[:, :, 3:4]                           # shape (224, 224, 1)
    elevation = img_array[:, :, 4:5]                    # shape (224, 224, 1)
    mask = img_array[:, :, 5:6]                         # shape (224, 224, 1)
    
    # Converting the elements into bytes
    rgb_bytes = image_rgb.tobytes()
    ir_bytes = ir.tobytes()
    elevation_bytes = elevation.tobytes()
    mask_bytes = mask.tobytes()
    file_name_bytes = os.path.basename(datapoint_path).encode('utf-8')
    
    
    feature = {
        'image_rgb': _bytes_feature(rgb_bytes),
        'IR': _bytes_feature(ir_bytes),
        'elevation': _bytes_feature(elevation_bytes),
        'mask': _bytes_feature(mask_bytes),
        'file_name': _bytes_feature(file_name_bytes),
        'lat': _float_feature(lat),
        'lon': _float_feature(lon),
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_datapoint(serialized_example):
    
    
    feature_description = {
        'image_rgb': tf.io.FixedLenFeature([], tf.string),  
        'IR': tf.io.FixedLenFeature([], tf.string),      
        'elevation': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'lat': tf.io.FixedLenFeature([], tf.float32),      
        'lon': tf.io.FixedLenFeature([], tf.float32)       
    }
    
    # Parsing the input example and returning to the original form
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    rgb = tf.io.decode_raw(example['image_rgb'], tf.float32)
    rgb = tf.reshape(rgb, (224, 224, 3))  
    rgb = tf.cast(rgb, tf.float32) / 255.0
    
    ir = tf.io.decode_raw(example['IR'], tf.float32)
    ir = tf.reshape(ir, (224, 224, 1))     
    
    elevation = tf.io.decode_raw(example['elevation'], tf.float32)
    elevation = tf.reshape(elevation, (224, 224, 1)) 
    
    # Rebuilding original image from channels
    image = tf.concat([rgb, ir, elevation], axis=-1)            # shape: (224, 224, 5)
    
    mask = tf.io.decode_raw(example['mask'], tf.float32)
    mask = tf.reshape(mask, (224, 224, 1)) 
    
    filename = example['file_name']
    
    lat = example['lat']
    lon = example['lon']
    
    return image, mask, filename, lat, lon


def parse_rgb_ir(serialized_datapoint):
    
    # Function to return only the RGB image and the IR image 
    image, mask, _, _, _ = parse_datapoint(serialized_datapoint)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image[:, :, :4], mask

def plot_history(history_json_path, figsize=(12, 4)):
    with open(history_json_path, 'r') as file:
        history = json.load(file)

    fig, axes = plt.subplots(1, 2, figsize=figsize) 

    # Plotting Loss vs Epochs
    axes[0].plot(history['loss'], label='Training Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='o')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plotting Accuracy vs Epochs
    axes[1].plot(history['accuracy'], label='Training Accuracy', marker='o')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    return history


def plot_inference_results(test_dataset, model, n_predictions):
    
    for images, labels in test_dataset.take(1): # Predicts one batch
        predictions = model.predict(images)
        
        true_labels = np.argmax(labels.numpy(), axis=-1)
        pred_labels = np.argmax(predictions, axis=-1)
        
        n = min(n_predictions, images.shape[0])
        
        fig, axes = plt.subplots(n, 4, figsize=(15, 5 * n))
        
        for i in range(n):
            
            axes[i, 0].imshow(images[i][:, :, :3])
            axes[i, 0].set_title("RGB Image")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true_labels[i], cmap='viridis', interpolation='none', vmin=0, vmax=5)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_labels[i], cmap='viridis', interpolation='none', vmin=0, vmax=5)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            h, w = true_labels[i].shape
            error_map = np.zeros((h, w, 3), dtype=np.uint8)
            correct_mask = (true_labels[i] == pred_labels[i])
            incorrect_mask = ~correct_mask
            
            error_map[correct_mask] = [0, 255, 0]                   # Green for correct
            error_map[incorrect_mask] = [255, 0, 0]                 # Red for incorrect
            
            axes[i, 3].imshow(error_map)
            axes[i, 3].set_title("Correct (Green) / Incorrect (Red)")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        

def parse_rgb(serialized_datapoint):
    # Function to parse inputs from TFRecord and return RGB image, IR image and Elevation and Mask
    image, mask, _, _, _ = parse_datapoint(serialized_datapoint)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image, mask


def build_datasets(train_filenames, val_filenames, test_filenames, parsing_function_train, parsing_function_val_test, batch_size=16):
    
    # Function to build train, val and test datasets from either a list of TFRecord filenames or from dataset objects
    
    if isinstance(train_filenames, list):               # A list of TFRecord file paths was passed as input
        train_dataset = (
            tf.data.TFRecordDataset(train_filenames)
            .map(parsing_function_train, num_parallel_calls=4)      
            .shuffle(buffer_size=100)                   
            .batch(batch_size)
            .prefetch(1)                                
        )
    else:           # A dataset object was passed as input
        train_dataset = train_filenames.map(parsing_function_train, num_parallel_calls=4).shuffle(buffer_size=100).batch(batch_size).prefetch(1) 

    if isinstance(val_filenames, list):
        val_dataset = (
            tf.data.TFRecordDataset(val_filenames)
            .map(parsing_function_val_test, num_parallel_calls=4)
            .batch(batch_size)
            .prefetch(1)
        )
    else:
        val_dataset = val_filenames.map(parsing_function_val_test, num_parallel_calls=4).batch(batch_size).prefetch(1)
    
    if isinstance(test_filenames, list):
        test_dataset = (
            tf.data.TFRecordDataset(test_filenames)
            .map(parsing_function_val_test, num_parallel_calls=4)
            .batch(batch_size)  
            .prefetch(1)
        )
    
    else:
        test_dataset = test_filenames.map(parsing_function_val_test, num_parallel_calls=4).batch(batch_size).prefetch(1)

    return train_dataset, val_dataset, test_dataset


def plot_datapoints_from_datasets(dataset_list, figsize, labels):

    # Function to plot datapoints belonging to the datasets in dataset_list according to their latitude and longitude
    
    colors = plt.cm.tab10.colors  
    plt.figure(figsize=figsize)
    
    for idx, dataset in enumerate(dataset_list):
        lats = []
        lons = []
        
        for record in dataset:
            _, _, filename, lat, lon = record
            lats.append(lat.numpy())
            lons.append(lon.numpy())
        
        lats = np.array(lats)
        lons = np.array(lons)
        
        plt.scatter(lons, lats, color=colors[idx % len(colors)], 
                    label=labels[idx], s=5, alpha=0.7)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Datapoints from Each TFRecord Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def split_datasets_by_grid(dataset_list, grid_size=0.005):

    # Function to take a list of datasets and create two datasets by separating in patches based on lat/lon
    
    min_lat = None
    min_lon = None
    for ds in dataset_list:
        for record in ds:           
            lat = record[3].numpy()
            lon = record[4].numpy()
            if min_lat is None or lat < min_lat:
                min_lat = lat
            if min_lon is None or lon < min_lon:
                min_lon = lon

    min_lat_tf = tf.constant(min_lat, dtype=tf.float32)
    min_lon_tf = tf.constant(min_lon, dtype=tf.float32)
    
    # We define two functions to determine whether a datapoint falls in the training or test region (i.e. is even or odd)
    def is_even(img, ir, el, lat, lon):
        cell_y = tf.floor((lat - min_lat_tf) / grid_size)
        cell_x = tf.floor((lon - min_lon_tf) / grid_size)
        parity = tf.cast(cell_x + cell_y, tf.int32) % 2
        return tf.equal(parity, 0)
    
    def is_odd(img, ir, el, lat, lon):
        cell_y = tf.floor((lat - min_lat_tf) / grid_size)
        cell_x = tf.floor((lon - min_lon_tf) / grid_size)
        parity = tf.cast(cell_x + cell_y, tf.int32) % 2
        return tf.equal(parity, 1)
    
    
    even_datasets = []
    odd_datasets = []
    
    # We split the dataset into odd or even
    for ds in dataset_list:
        even_datasets.append(ds.filter(is_even))
        odd_datasets.append(ds.filter(is_odd))
    
    # In case more than one dataset was passed as input, merge even datasets and odd datasets
    dataset_even = even_datasets[0]
    for ds in even_datasets[1:]:
        dataset_even = dataset_even.concatenate(ds)
    
    dataset_odd = odd_datasets[0]
    for ds in odd_datasets[1:]:
        dataset_odd = dataset_odd.concatenate(ds)
    
    return dataset_even, dataset_odd


def split_train_val(train_dataset, split_ratio=0.8, shuffle_buffer=100, seed=1):

    # Function to split a dataset into train and val according to the split ratio.
    train_dataset = train_dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)
    
    total_examples = 0
    for _ in train_dataset:
        total_examples += 1

    train_count = int(total_examples * split_ratio)
    train_ds = train_dataset.take(train_count)
    val_ds = train_dataset.skip(train_count)
    
    return train_ds, val_ds

def parse_easy(image, mask, name, lat, lon):
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image, mask

def plot_val_and_loss(validation_scores, test_scores, model_names):

    x = np.arange(len(model_names))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, validation_scores, width, label='Validation', color='skyblue')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Test', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Validation vs Test Scores by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Function to add the score on top of the bar to make plots nicer
    def autolabel(rects):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()
    

def augment(image, mask):
    
    # Function to do data augmentation during training.
    
    # Apply a random rotation of a multiple of 90 degrees
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Apply a horizontal flip with probability 1/2
    flip_lr = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip_lr, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # Apply a random vertical flip with probability 1/2
    flip_ud = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip_ud, lambda: tf.image.flip_up_down(mask), lambda: mask)

    return image, mask


def parse_rgb_ir_augmented(serialized_datapoint):
    # Function to return the image after augmentation
    
    image, mask = parse_rgb_ir(serialized_datapoint)
    image, mask = augment(image, mask)
    
    return image, mask

def parse_rgb_augmented(serialized_datapoint):
    # Function to return the image after augmentation
    image, mask = parse_rgb(serialized_datapoint)
    image, mask = augment(image, mask)
    
    return image, mask
