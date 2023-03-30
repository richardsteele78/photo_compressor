# streamlit run "C:\Users\RichardSteele\Documents\Python Scripts\streamlit_photocompressor.py"
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st

st.header("📷 K-MEANS COMPRESSION")
img_file_buffer = st.camera_input("Take a picture")
num_of_centroids = st.slider(label="Select number of KMEANS CLUSTERS to compress image",min_value=2,max_value=20,value=4)

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    im = Image.open(img_file_buffer)
    # get pixels of the image
    pixel_np = np.asarray(im)
    # reshape array (remove rows and columns)
    image_height = im.height
    image_width = im.width
    pixel_np = np.reshape(pixel_np, (image_height * image_width, 3))
    # run k-means clustering on the pixel data
    compressor = KMeans(n_clusters=num_of_centroids)
    compressor.fit(pixel_np)
    # create an array replacing each pixel label with its corresponding cluster centroid
    pixel_centroid = np.array([list(compressor.cluster_centers_[label]) for label in compressor.labels_])
    # convert the array to an unsigned integer type
    pixel_centroid = pixel_centroid.astype("uint8")
    # reshape this array according to the height and width of our image
    pixel_centroids_reshaped = np.reshape(pixel_centroid, (image_height, image_width, 3), "C")
    # create the compressed image
    compressed_im = Image.fromarray(pixel_centroids_reshaped)
    st.image(compressed_im)
    st.header("RGB Codes of " + str(num_of_centroids) + " colours")
    st.write(compressor.cluster_centers_)