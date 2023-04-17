# streamlit run "C:\Users\RichardSteele\Documents\Python Scripts\Image\streamlit_photocompressor.py"

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st
#import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

st.header("📷 K-MEANS PHOTO COMPRESSION")
img_file_buffer = st.camera_input("Generate a dataset by taking a photograph")
num_of_centroids = st.slider(label="Select number of colours (using K-MEANS clustering) to compress image",min_value=2,max_value=20,value=4)

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
    st.header("RGB Codes of " + str(num_of_centroids) + " colour clusters")
    #df['Red'] = compressor.cluster_centers_
    fulldf = pd.DataFrame(data=pixel_np , columns=['RED','GREEN','BLUE'])
    fulldf['RGBCODE'] =  tuple(zip(fulldf["RED"]/255., fulldf["GREEN"]/255., fulldf["BLUE"]/255.))
    df = pd.DataFrame(data=compressor.cluster_centers_ , columns=['RED','GREEN','BLUE'])
    df['RGBCODE'] =  tuple(zip(df["RED"]/255., df["GREEN"]/255., df["BLUE"]/255.))
    st.write(df)
    #fig1 = px.scatter_3d(df, x='RED', y='GREEN', z='BLUE')
    #st.plotly_chart(fig1)
    
    fullscatter = st.button("Load full pixel scatter (with clusters)? (takes a few seconds)")
    if fullscatter:
        fig2 = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig2.add_subplot(projection='3d')
        #ax = plt.subplot(111,aspect = 'equal')
        # Creating plot
        fullcolors=np.array(fulldf['RGBCODE'])
        ax.scatter(xs=fulldf['RED'], ys=fulldf['GREEN'], zs=fulldf['BLUE'],s = 3,alpha=0.01, c = fullcolors)
        x=df['RED']
        y=df['GREEN']
        z=df['BLUE']
        colors=np.array(df['RGBCODE'])
        ax.scatter(xs=x, ys=y, zs=z,s = 100,alpha=1,c=colors,edgecolors="black")
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B',labelpad=0.50)
        ax.set_title("Scatter plot of pixels in 3D (RGB) space")
        st.pyplot(fig2)
        st.caption("Note that the diagonal betweeen (0,0,0) and (255,255,255) corresponds the greyscale diagonal between white and black")
        
    loadelbow = st.button("Evaluate Elbow plot? (takes around 1-2 minutes!)")
    if loadelbow:
        my_bar = st.progress(0)
        #PLOT ELBOW
        inertia = []
        for i in range(2,21):
            compressor = fit_kmeans_model(pixel_np,i)
            inertia.append(compressor.inertia_)
            my_bar.progress((i-1) / 19)
        fig3 = plt.figure()
        ax = fig3.add_subplot()
        # Creating plot
        ax.scatter(x=np.arange(1,20), y=inertia)
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Inertia')
        st.pyplot(fig3)
