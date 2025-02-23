import rasterio as rio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.transform import xy
import numpy as np
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

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


#def serialize_datapoint(datapoint_path):
    
    # # Reading .tiff file
    # img_array = read_tiff(datapoint_path)
    
    
    # # Splitting into RGB, IR, Elevation and Segmentation Mask
    # image_rgb = img_array[:, :, :3]             # (224, 224, 3)
    # ir = img_array[:, :, 3:4]                   # (224, 224, 1)
    # elevation = img_array[:, :, 4:5]            # (224, 224, 1)
    # mask = img_array[:, :, 5:6]                 # (224, 224, 1)
    
    # # Converting to bytes
    # rgb_bytes = image_rgb.tobytes()
    # ir_bytes = ir.tobytes()
    # elevation_bytes = elevation.tobytes()
    # mask_bytes = mask.tobytes()
    # file_name_bytes = os.path.basename(datapoint_path).encode('utf-8')
    
    # # Creating feature dict
    # feature = {
    #     'image_rgb': _bytes_feature(rgb_bytes),
    #     'IR': _bytes_feature(ir_bytes),
    #     'elevation': _bytes_feature(elevation_bytes),
    #     'mask': _bytes_feature(mask_bytes),
    #     'file_name': _bytes_feature(file_name_bytes)
    # }
    
    # # Creating single tf.example
    # example = tf.train.Example(features=tf.train.Features(feature=feature))
    # return example.SerializeToString()

def serialize_datapoint(datapoint_path):
    # Read the TIFF file (assuming your read_tiff returns a NumPy array)
    img_array = read_tiff(datapoint_path)
    
    # Compute the center latitude and longitude using Rasterio.
    with rio.open(datapoint_path) as dataset:
        width, height = dataset.width, dataset.height
        center_x = width // 2
        center_y = height // 2
        # rasterio.transform.xy returns (lon, lat)
        lon, lat = xy(dataset.transform, center_y, center_x)
    
    # Split the image into RGB, IR, Elevation, and Mask channels.
    image_rgb = img_array[:, :, :3]      # shape (224, 224, 3)
    ir = img_array[:, :, 3:4]             # shape (224, 224, 1)
    elevation = img_array[:, :, 4:5]      # shape (224, 224, 1)
    mask = img_array[:, :, 5:6]           # shape (224, 224, 1)
    
    # Convert the arrays to bytes.
    rgb_bytes = image_rgb.tobytes()
    ir_bytes = ir.tobytes()
    elevation_bytes = elevation.tobytes()
    mask_bytes = mask.tobytes()
    file_name_bytes = os.path.basename(datapoint_path).encode('utf-8')
    
    # Create a dictionary mapping the feature names to the tf.train.Example-compatible data types.
    feature = {
        'image_rgb': _bytes_feature(rgb_bytes),
        'IR': _bytes_feature(ir_bytes),
        'elevation': _bytes_feature(elevation_bytes),
        'mask': _bytes_feature(mask_bytes),
        'file_name': _bytes_feature(file_name_bytes),
        'lat': _float_feature(lat),
        'lon': _float_feature(lon),
    }
    
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# def parse_datapoint(serialized_example):
    
#     feature_description = {
#         'image_rgb': tf.io.FixedLenFeature([], tf.string),  
#         'IR': tf.io.FixedLenFeature([], tf.string),      
#         'elevation': tf.io.FixedLenFeature([], tf.string),
#         'mask': tf.io.FixedLenFeature([], tf.string),
#         'file_name': tf.io.FixedLenFeature([], tf.string)
#     }
    
#     example = tf.io.parse_single_example(serialized_example, feature_description)
    
#     rgb = tf.io.decode_raw(example['image_rgb'], tf.float32)
#     rgb = tf.reshape(rgb, (224, 224, 3))  
#     rgb = tf.cast(rgb, tf.float32) / 255.0
    
#     ir = tf.io.decode_raw(example['IR'], tf.float32)
#     ir = tf.reshape(ir, (224, 224, 1))     
    
#     elevation = tf.io.decode_raw(example['elevation'], tf.float32)
#     elevation = tf.reshape(elevation, (224, 224, 1)) 
    
#     # Rebuilding original image
#     image = tf.concat([rgb, ir, elevation], axis=-1)  
    
    
#     mask = tf.io.decode_raw(example['mask'], tf.float32)
#     mask = tf.reshape(mask, (224, 224, 1)) 
    
#     # Extract the label.
#     filename = example['file_name']
    
#     return image, mask, filename 

def parse_datapoint(serialized_example):
    
    feature_description = {
        'image_rgb': tf.io.FixedLenFeature([], tf.string),  
        'IR': tf.io.FixedLenFeature([], tf.string),      
        'elevation': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'lat': tf.io.FixedLenFeature([], tf.float32),      # Added latitude feature
        'lon': tf.io.FixedLenFeature([], tf.float32)       # Added longitude feature
    }
    
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    rgb = tf.io.decode_raw(example['image_rgb'], tf.float32)
    rgb = tf.reshape(rgb, (224, 224, 3))  
    rgb = tf.cast(rgb, tf.float32) / 255.0
    
    ir = tf.io.decode_raw(example['IR'], tf.float32)
    ir = tf.reshape(ir, (224, 224, 1))     
    
    elevation = tf.io.decode_raw(example['elevation'], tf.float32)
    elevation = tf.reshape(elevation, (224, 224, 1)) 
    
    # Rebuilding original image from channels.
    image = tf.concat([rgb, ir, elevation], axis=-1)  
    
    mask = tf.io.decode_raw(example['mask'], tf.float32)
    mask = tf.reshape(mask, (224, 224, 1)) 
    
    filename = example['file_name']
    
    # Extract the latitude and longitude.
    lat = example['lat']
    lon = example['lon']
    
    return image, mask, filename, lat, lon



# def parse_rgb_ir(serialized_datapoint):
#     image, mask, _ = parse_datapoint(serialized_datapoint)
#     mask = tf.squeeze(mask, axis=-1)
#     mask = tf.cast(mask, tf.int32)
#     mask = tf.one_hot(mask, depth=6)
#     return image[:, :, :4], mask

def parse_rgb_ir(serialized_datapoint):
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
        
        
# def parse_rgb(serialized_datapoint):
#     image, mask, _ = parse_datapoint(serialized_datapoint)
#     mask = tf.squeeze(mask, axis=-1)
#     mask = tf.cast(mask, tf.int32)
#     mask = tf.one_hot(mask, depth=6)
#     return image, mask

def parse_rgb(serialized_datapoint):
    image, mask, _, _, _ = parse_datapoint(serialized_datapoint)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image, mask


def build_datasets(train_filenames, val_filenames, test_filenames, parsing_function, batch_size=16):
    
    if isinstance(train_filenames, list):
        train_dataset = (
            tf.data.TFRecordDataset(train_filenames)
            .map(parsing_function, num_parallel_calls=4)      
            .shuffle(buffer_size=100)                   
            .batch(batch_size)
            .prefetch(1)                                
        )
    else:
        train_dataset = train_filenames.map(parsing_function, num_parallel_calls=4).shuffle(buffer_size=100).batch(batch_size).prefetch(1) 

    if isinstance(val_filenames, list):
        val_dataset = (
            tf.data.TFRecordDataset(val_filenames)
            .map(parsing_function, num_parallel_calls=4)
            .batch(batch_size)
            .prefetch(1)
        )
    else:
        val_dataset = val_filenames.map(parsing_function, num_parallel_calls=4).batch(batch_size).prefetch(1)
    
    if isinstance(test_filenames, list):
        test_dataset = (
            tf.data.TFRecordDataset(test_filenames)
            .map(parsing_function, num_parallel_calls=4)
            .batch(batch_size)  
            .prefetch(1)
        )
    
    else:
        test_dataset = test_filenames.map(parsing_function, num_parallel_calls=4).batch(batch_size).prefetch(1)

    return train_dataset, val_dataset, test_dataset


def plot_datapoints_from_datasets(dataset_list, figsize):

    colors = plt.cm.tab10.colors  # A palette of distinct colors.
    plt.figure(figsize=figsize)
    
    for idx, dataset in enumerate(dataset_list):
        lats = []
        lons = []
        
        # Iterate over the dataset and collect latitudes and longitudes.
        for record in dataset:
            _, _, filename, lat, lon = record
            lats.append(lat.numpy())
            lons.append(lon.numpy())
        
        lats = np.array(lats)
        lons = np.array(lons)
        
        # Plot the points using a unique color per dataset.
        plt.scatter(lons, lats, color=colors[idx % len(colors)], 
                    label=f"Dataset {idx+1}", s=5, alpha=0.7)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Datapoints from Each TFRecord Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def split_datasets_by_grid(dataset_list, grid_size=0.005):

    # First pass: compute global minimum latitude and longitude across all datasets.
    # Note: Iterating over the dataset will consume it, so consider caching in practice.
    min_lat = None
    min_lon = None
    for ds in dataset_list:
        for record in ds:
            # record = (image, mask, filename, lat, lon)
            lat = record[3].numpy()
            lon = record[4].numpy()
            if min_lat is None or lat < min_lat:
                min_lat = lat
            if min_lon is None or lon < min_lon:
                min_lon = lon

    # Convert to TensorFlow constants for use in our filter functions.
    min_lat_tf = tf.constant(min_lat, dtype=tf.float32)
    min_lon_tf = tf.constant(min_lon, dtype=tf.float32)
    
    # Define predicate functions that accept five arguments.
    def is_even(image, mask, filename, lat, lon):
        cell_y = tf.floor((lat - min_lat_tf) / grid_size)
        cell_x = tf.floor((lon - min_lon_tf) / grid_size)
        parity = tf.cast(cell_x + cell_y, tf.int32) % 2
        return tf.equal(parity, 0)
    
    def is_odd(image, mask, filename, lat, lon):
        cell_y = tf.floor((lat - min_lat_tf) / grid_size)
        cell_x = tf.floor((lon - min_lon_tf) / grid_size)
        parity = tf.cast(cell_x + cell_y, tf.int32) % 2
        return tf.equal(parity, 1)
    
    # For each input dataset, filter into even and odd records.
    even_datasets = []
    odd_datasets = []
    for ds in dataset_list:
        even_datasets.append(ds.filter(is_even))
        odd_datasets.append(ds.filter(is_odd))
    
    # Concatenate the individual datasets into one even and one odd dataset.
    dataset_even = even_datasets[0]
    for ds in even_datasets[1:]:
        dataset_even = dataset_even.concatenate(ds)
    
    dataset_odd = odd_datasets[0]
    for ds in odd_datasets[1:]:
        dataset_odd = dataset_odd.concatenate(ds)
    
    return dataset_even, dataset_odd

def split_train_val(train_dataset, split_ratio=0.8, shuffle_buffer=100, seed=42):

    train_dataset = train_dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)
    
    total_examples = 0
    for _ in train_dataset:
        total_examples += 1

    train_count = int(total_examples * split_ratio)
    
    # Split the dataset using take and skip.
    train_ds = train_dataset.take(train_count)
    val_ds = train_dataset.skip(train_count)
    
    return train_ds, val_ds

def parse_easy(image, mask, name, lat, lon):
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=6)
    return image, mask