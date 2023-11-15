import streamlit as st
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
st.set_page_config(
    page_title="Explore The Dataset",
    page_icon="📊",
    layout="centered",
    )
@st.cache_data
def fetch_and_clean_data(url):
    X, y = fetch_openml(url, version=1, return_X_y=True, parser="pandas")
    X=np.array(X)
    y=np.array(y)
    return X,y
X,y= fetch_and_clean_data("mnist_784")

st.title("Explore the Dataset: Dive into the World of MNIST")

st.write("Welcome to the heart of our Exploration Room – the place where you get hands-on with the MNIST dataset! 📊")

st.write("#### Slide Through the Samples")
st.write("Use the slider below to explore the MNIST dataset. Select the number of samples you want to see!")
n=st.select_slider("Number of samples :",(1,4,9,16,25,36))
n=int(n)
p=np.random.randint(y.shape[0],size=n)

f=plt.figure(figsize=(10, 10)) 
x=int(np.sqrt(n))

for i in range(x):  
    for j in range(int(n/x)): 
        index = i * x + j
        plt.subplot(x, int(n/x), index + 1) 
        plt.imshow(X[p[index]].reshape(28, 28))  # Display the image
        plt.xticks([])  # Remove x-ticks
        plt.yticks([])  # Remove y-ticks
        plt.title(y[p[index]])  # Display the label as title with reduced font size

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
plt.tight_layout()  # Adjust the spacing between plots for better visualization
st.pyplot(f)  # Display the entire grid
st.write("#### Test-Train Split Wizard")
st.write("Now, let's empower you to create your own dataset using the Test-Train Split Wizard. Choose the number of test samples you want to reserve and set the split ratio to customize your dataset. This is where you take control of the learning experience!")
n_test = st.slider("Test Samples",min_value=1000,max_value=10000,step=1000)
split_ratio=st.slider("Split Ratio",min_value=0.1,max_value=0.9,step=0.1)
n_train = int(n_test*split_ratio/(1-split_ratio))

split_loc = 35000 # train and test split at location of 60k
X_train, y_train = X[:n_train,:], y[:n_train]
X_test, y_test = X[split_loc:split_loc + n_test, :] , y[split_loc:split_loc+n_test]
df={"Train Data" : n_train, "Test Data": n_test}
st.table(df)
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(data = y_train, columns = ['class'])
y_test_df = pd.DataFrame(data = y_test, columns = ['class'])
f1, axs = plt.subplots(1, 2, figsize=(10,5))
y_train_df['class'].value_counts().plot(kind = 'bar', colormap = 'Paired',ax=axs[0])
axs[0].set_xlabel('Class')
axs[0].set_ylabel('Number of samples for each category')
axs[0].set_title('Training set')
y_test_df['class'].value_counts().plot(kind = 'bar', colormap = 'Paired',ax=axs[1])
axs[1].set_xlabel('Class')
axs[1].set_ylabel('Number of samples for each category')
axs[1].set_title('Testing set')
st.pyplot(f1)
if "train_data" not in st.session_state:
    st.session_state["train_data"]=n_train
if "test_data" not in st.session_state:
    st.session_state["test_data"]=n_test
#st.write(st.session_state["train_data"])  
#st.write(st.session_state["test_data"])  

