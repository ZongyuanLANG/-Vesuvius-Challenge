#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_addons.metrics as metrics
from tensorflow.keras import layers
import numpy as np
import glob
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


data_dir = 'vesuvius-challenge-ink-detection/'
BUFFER = 32  # Half-size of papyrus patches we'll use as model inputs
Z_DIM = 20   # Number of slices in the z direction. Max value is 64 - Z_START
Z_START = 16  ## offset
SHARED_HEIGHT = 2000 

# Model config
bs = 16


# In[ ]:





# In[ ]:


def resize(img):
    current_width, current_height = img.size
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    img = img.resize(new_size)
    return img

def load_mask(split, index):
    img = Image.open(f"{data_dir}/{split}/{index}/mask.png").convert('1')
    img = resize(img)
    return tf.convert_to_tensor(np.array(img), dtype="bool")

def load_labels(split, index):
    img = Image.open(f"{data_dir}/{split}/{index}/inklabels.png")
    img = resize(img)
    return tf.convert_to_tensor(np.array(img), dtype="bool")

mask = load_mask(split="train", index=1)
labels = load_labels(split="train", index=1)



mask_test_a = load_mask(split="test", index="a")
mask_test_b = load_mask(split="test", index="b")

mask_train_1 = load_mask(split="train", index=1)
labels_train_1 = load_labels(split="train", index=1)

mask_train_2 = load_mask(split="train", index=2)
labels_train_2 = load_labels(split="train", index=2)

mask_train_3 = load_mask(split="train", index=3)
labels_train_3 = load_labels(split="train", index=3)


# In[ ]:


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    z_slices_fnames = sorted(glob.glob(f"{data_dir}/{split}/{index}/surface_volume/*.tif"))[Z_START:Z_START + Z_DIM]
    z_slices = []
    for z, filename in  enumerate(z_slices_fnames):
        img = Image.open(filename)
        img = resize(img)
        z_slice = np.array(img, dtype="float32")
        z_slices.append(z_slice)
    return tf.stack(z_slices, axis=-1)


# In[ ]:


volume_train_1 = load_volume(split="train", index=1)
volume_train_2 = load_volume(split="train", index=2)
volume_train_3 = load_volume(split="train", index=3)
volume = tf.concat([volume_train_1, volume_train_2, volume_train_3], axis=1)
labels = tf.concat([labels_train_1, labels_train_2, labels_train_3], axis=1)
mask = tf.concat([mask_train_1, mask_train_2, mask_train_3], axis=1)


# In[ ]:





# In[ ]:


val_location = (1300//2, 1000//2)
val_zone_size = (600//2, 2000//2)


# In[ ]:


def sample_random_location(shape):
    random_train_x = tf.random.uniform(shape=(), minval=BUFFER, maxval=shape[0] - BUFFER - 1, dtype="int32")
    random_train_y = tf.random.uniform(shape=(), minval=BUFFER, maxval=shape[1] - BUFFER - 1, dtype="int32")
    random_train_location = tf.stack([random_train_x, random_train_y])
    return random_train_location

def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]]

sample_random_location_train = lambda x: sample_random_location(mask.shape)
is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

def is_in_val_zone(location, val_location, val_zone_size):
    x = location[0]
    y = location[1]
    x_match = val_location[0] - BUFFER <= x <= val_location[0] + val_zone_size[0] + BUFFER
    y_match = val_location[1] - BUFFER <= y <= val_location[1] + val_zone_size[1] + BUFFER
    return x_match and y_match

def is_proper_train_location(location):
    return not is_in_val_zone(location, val_location, val_zone_size) and is_in_mask_train(location)

train_locations_ds = tf.data.Dataset.from_tensor_slices([0]).repeat().map(sample_random_location_train, num_parallel_calls=tf.data.AUTOTUNE)
train_locations_ds = train_locations_ds.filter(is_proper_train_location)


# In[ ]:





# In[ ]:


def extract_subvolume(location, volume):
    x = location[0]
    y = location[1]
    subvolume = volume[x-BUFFER:x+BUFFER, y-BUFFER:y+BUFFER, :]
    subvolume = tf.cast(subvolume, dtype="float32") / 65535.
    return subvolume

def extract_labels(location, labels):
    x = location[0]
    y = location[1]
    label = labels[x-BUFFER:x+BUFFER, y-BUFFER:y+BUFFER]
    label = tf.cast(label, dtype="float32")
    label = tf.expand_dims(label, axis=-1)
    return label

def extract_subvolume_and_label(location):
    subvolume = extract_subvolume(location, volume)
    label = extract_labels(location, labels)
    return subvolume, label

shuffle_buffer_size = bs * 4

train_ds = train_locations_ds.map(extract_subvolume_and_label, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE).batch(bs)


# In[ ]:




val_locations_stride = BUFFER
val_locations = []
for x in range(val_location[0], val_location[0] + val_zone_size[0], val_locations_stride):
    for y in range(val_location[1], val_location[1] + val_zone_size[1], val_locations_stride):
        val_locations.append((x, y))

val_locations_ds = tf.data.Dataset.from_tensor_slices(val_locations).filter(is_in_mask_train)
val_ds = val_locations_ds.map(extract_subvolume_and_label, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE).batch(bs)


# In[ ]:


augmenter = keras.Sequential([
    layers.RandomContrast(0.2),
])

def augment_train_data(data, label):
    data = augmenter(data)
    return data, label

augmented_train_ds = train_ds.map(augment_train_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# In[ ]:





# In[ ]:


def get_model(input_shape):
    inputs = keras.Input(input_shape)
    
    x = inputs
    print(x.shape)
    filters = 64
    
    ## entry block
    
    x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    skip = x
    
    ## first block
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x,skip])
    x = layers.Activation("relu")(x)
    filters *= 2
    x1 = x
    x = layers.Conv2D(filters = filters,kernel_size = 1,strides = 2,padding = 'same')(x)
    
    
    skip = x
    
    ## second block
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x,skip])
    x = layers.Activation("relu")(x)
    filters*=2
    x2 = x
    x = layers.Conv2D(filters = filters,kernel_size = 1,strides = 2,padding = 'same')(x)
    
    skip = x

    ##third block
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x,skip])
    x = layers.Activation("relu")(x)
    
    x3 = x
    skip = x
    skip = layers.Conv2D(filters = filters,kernel_size = 1,strides = 2,padding = 'same')(skip)
    
    ## last block
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters = filters,kernel_size = 3,padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x4 = layers.add([x,skip])
    
    
    ## upsampling
    out4 = layers.UpSampling2D(16)(x4)
    skip3 = layers.UpSampling2D(2)(x4)
    out3 = layers.add([x3,skip3])
    skip2 = layers.UpSampling2D(2)(out3)
    out3 = layers.UpSampling2D(8)(out3)
    x2 = layers.Conv2D(filters = filters,kernel_size = 1,padding = 'same')(x2)
    out2 = layers.add([x2,skip2])
    skip1 = layers.UpSampling2D(2)(out2)
    out2 = layers.UpSampling2D(4)(out2)
    x1 = layers.Conv2D(filters = filters,kernel_size = 1,padding = 'same')(x1)
    out1 = layers.add([x1,skip1])
    out1 = layers.UpSampling2D(2)(out1)
    
    ##prediction
    out = layers.add([out1,out2,out3,out4])
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(out)
    
    model = keras.Model(inputs, out)
    return model


model = get_model((BUFFER * 2, BUFFER * 2, Z_DIM))
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

history = model.fit(augmented_train_ds, validation_data=val_ds, epochs=30,steps_per_epoch = 1000)
model.save("model.keras")


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.close()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc.png')
plt.close()


# In[ ]:


def metric(dataset,model,beta = 0.5):
    beta_square = beta**2
    total_TP = 0
    total_FP = 0
    total_true = 0
    for imgs, batch_label in dataset:
        pred = np.array(model.predict(imgs))
        pred = pred>0.5
        
        TP = pred[batch_label == 1].sum()
        FP = pred[batch_label == 0].sum()
        trues = np.array(batch_label).sum()
        
        total_TP+= TP
        total_FP += FP
        total_true += trues
    precision = total_TP/(total_TP+total_FP)
    recall = total_TP/(total_true)
    Fbeta = (1+beta_square)*(precision*recall)/(beta_square*precision + recall)
    return recall,precision,Fbeta
model = keras.models.load_model("model.keras")
Recall,Precision,F_beta = metric(val_ds,model,0.5)
print(f"Recall: {Recall}\nPrecision:{Precision}\nF_beta:{F_beta} ")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def compute_predictions_map(split, index):
    print(f"Load data for {split}/{index}")

    test_volume = load_volume(split=split, index=index)
    test_mask = load_mask(split=split, index=index)

    test_locations = []
    stride = BUFFER // 2
    for x in range(BUFFER, test_volume.shape[0] - BUFFER, stride):
        for y in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
            test_locations.append((x, y))

    print(f"{len(test_locations)} test locations (before filtering by mask)")

    sample_random_location_test = lambda x: sample_random_location(test_mask.shape)
    is_in_mask_test = lambda x: is_in_masked_zone(x, test_mask)
    extract_subvolume_test = lambda x: extract_subvolume(x, test_volume)

    test_locations_ds = tf.data.Dataset.from_tensor_slices(test_locations).filter(is_in_mask_test)
    test_ds = test_locations_ds.map(extract_subvolume_test, num_parallel_calls=tf.data.AUTOTUNE)

    predictions_map = np.zeros(test_volume.shape[:2] + (1,), dtype="float16")
    predictions_map_counts = np.zeros(test_volume.shape[:2] + (1,), dtype="int8")

    print(f"Compute predictions")

    for loc_batch, patch_batch in zip(test_locations_ds.batch(bs), test_ds.batch(bs)):
        predictions = model.predict_on_batch(patch_batch)
        for (x, y), pred in zip(loc_batch, predictions):
            predictions_map[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :] += pred
            predictions_map_counts[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :] += 1  
    predictions_map /= (predictions_map_counts + 1e-7)
    return predictions_map


# In[ ]:


predictions_map_a = compute_predictions_map(split="test", index="a")
predictions_map_b = compute_predictions_map(split="test", index="b")
predictions_map_train = compute_predictions_map(split="train", index="1")


# In[ ]:


plt.imsave('train.png', predictions_map_train.squeeze()>0.1,cmap='gray')
plt.imsave('test_a.png',predictions_map_a.suqeeze()>0.1,cmap = 'gray')
plt.imsave('test_b.png',predictions_map_b.squeeze()>0.1,cmap = 'gray')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




