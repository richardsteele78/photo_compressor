# streamlit run "C:\Users\RichardSteele\Documents\Python Scripts\Image\streamlit_photocompressor.py"

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
 
st.header("📷 K-MEANS PHOTO COMPRESSION")
img_file_buffer = st.camera_input("Take a picture")
num_of_centroids = st.slider(label="Select number of colours (KMEANS CLUSTERS) to compress image",min_value=2,max_value=20,value=4)

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
    @st.cache_data
    def fit_kmeans_model(pixel_np,num_of_centroids):
        compressor = KMeans(n_clusters=num_of_centroids)
        compressor.fit(pixel_np)
        #st.balloons()
        return compressor
    compressor = fit_kmeans_model(pixel_np,num_of_centroids)
    # create an array replacing each pixel label with its corresponding cluster centroid
    pixel_centroid = np.array([list(compressor.cluster_centers_[label]) for label in compressor.labels_])
    # convert the array to an unsigned integer type
    pixel_centroid = pixel_centroid.astype("uint8")
    # reshape this array according to the height and width of our image
    pixel_centroids_reshaped = np.reshape(pixel_centroid, (image_height, image_width, 3), "C")
    # create the compressed image
    @st.cache_data
    def compress_image(pixel_centroids_reshaped):
        compressed_im = Image.fromarray(pixel_centroids_reshaped)
        st.image(compressed_im)
        #st.balloons()
        return compressed_im
    compress_image(pixel_centroids_reshaped)
    st.header("RGB Codes of " + str(num_of_centroids) + " colours")
    #df['Red'] = compressor.cluster_centers_
    fulldf = pd.DataFrame(data=pixel_np , columns=['RED','GREEN','BLUE'])
    df = pd.DataFrame(data=compressor.cluster_centers_ , columns=['RED','GREEN','BLUE'])
    df['RGBCODE'] = '(' + df["RED"].astype(str) + ',' + df["GREEN"].astype(str) + ',' + df["BLUE"].astype(str) + ')'
    st.write(df)
    fig1 = px.scatter_3d(df, x='RED', y='GREEN', z='BLUE')
    st.plotly_chart(fig1)
    fullscatter = st.button("Load full pixel scatter? (takes a few seconds)")
    if fullscatter:
#        fig2 = px.scatter_3d(fulldf, x='RED', y='GREEN', z='BLUE')
#        fig2.update_traces(marker_size = 1)
#        st.plotly_chart(fig2)
        fig3 = plt.figure()
        ax = fig3.add_subplot(projection='3d')
        # Creating plot
        ax.scatter(xs=fulldf['RED'], ys=fulldf['GREEN'], zs=fulldf['BLUE'],s = 1,alpha=0.01, color = "gray")
        ax.scatter(xs=df['RED'], ys=df['GREEN'], zs=df['BLUE'],s = 50,alpha=1,c="black") #, color = "green")
        st.pyplot(fig3)
