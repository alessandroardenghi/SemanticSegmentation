import rasterio as rio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import json

def read_tiff(datapoint_path):
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
    2.0: 'Low Vegetation',
    3.0: 'Tree',
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
    

def _bytes_feature(value):
    
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # Convert tensor to bytes.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_datapoint(datapoint_path):
    
    # Reading .tiff file
    img_array = read_tiff(datapoint_path)
    
    
    # Splitting into RGB, IR, Elevation and Segmentation Mask
    image_rgb = img_array[:, :, :3]             # (224, 224, 3)
    ir = img_array[:, :, 3:4]                   # (224, 224, 1)
    elevation = img_array[:, :, 4:5]            # (224, 224, 1)
    mask = img_array[:, :, 5:6]                 # (224, 224, 1)
    
    # Converting to bytes
    rgb_bytes = image_rgb.tobytes()
    ir_bytes = ir.tobytes()
    elevation_bytes = elevation.tobytes()
    mask_bytes = mask.tobytes()
    file_name_bytes = os.path.basename(datapoint_path).encode('utf-8')
    
    # Creating feature dict
    feature = {
        'image_rgb': _bytes_feature(rgb_bytes),
        'IR': _bytes_feature(ir_bytes),
        'elevation': _bytes_feature(elevation_bytes),
        'mask': _bytes_feature(mask_bytes),
        'file_name': _bytes_feature(file_name_bytes)
    }
    
    # Creating single tf.example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_datapoint(serialized_example):
    
    feature_description = {
        'image_rgb': tf.io.FixedLenFeature([], tf.string),  
        'IR': tf.io.FixedLenFeature([], tf.string),      
        'elevation': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'file_name': tf.io.FixedLenFeature([], tf.string)
    }
    
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    rgb = tf.io.decode_raw(example['image_rgb'], tf.float32)
    rgb = tf.reshape(rgb, (224, 224, 3))  
    rgb = tf.cast(rgb, tf.float32) / 255.0
    
    ir = tf.io.decode_raw(example['IR'], tf.float32)
    ir = tf.reshape(ir, (224, 224, 1))     
    
    elevation = tf.io.decode_raw(example['elevation'], tf.float32)
    elevation = tf.reshape(elevation, (224, 224, 1)) 
    
    # Rebuilding original image
    image = tf.concat([rgb, ir, elevation], axis=-1)  
    
    
    mask = tf.io.decode_raw(example['mask'], tf.float32)
    mask = tf.reshape(mask, (224, 224, 1)) 
    
    # Extract the label.
    filename = example['file_name']
    
    return image, mask, filename 


def parse_rgb_ir(serialized_datapoint):
    image, mask, _ = parse_datapoint(serialized_datapoint)
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
    for images, labels in test_dataset.take(1):
        
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
            
            error_map[correct_mask] = [0, 255, 0]    # Green for correct
            error_map[incorrect_mask] = [255, 0, 0]    # Red for incorrect
            
            axes[i, 3].imshow(error_map)
            axes[i, 3].set_title("Correct (Green) / Incorrect (Red)")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
def parse_rgb(serialized_datapoint):
    image, mask, _ = parse_datapoint(serialized_datapoint)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image, mask


def build_datasets(train_filenames, val_filenames, test_filenames, parsing_function, batch_size=16):
    train_dataset = (
        tf.data.TFRecordDataset(train_filenames)
        .map(parsing_function, num_parallel_calls=4)      
        .shuffle(buffer_size=100)                   
        .batch(batch_size)
        .prefetch(1)                                
    )

    val_dataset = (
        tf.data.TFRecordDataset(val_filenames)
        .map(parsing_function, num_parallel_calls=4)
        .batch(batch_size)
        .prefetch(1)
    )

    test_dataset = (
        tf.data.TFRecordDataset(test_filenames)
        .map(parsing_function, num_parallel_calls=4)
        .batch(batch_size)  # Use the same batch size as during training.
        .prefetch(1)
    )

    return train_dataset, val_dataset, test_dataset