import streamlit as st
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
import visualkeras
st.set_page_config(
        page_title="Neural Network Adventure",
        page_icon="🧠",
        layout="centered",
    )

# Section 1: Design Your Neural Network
st.title("Welcome to the Neural Network Adventure! 🚀")
st.write("""Embark on an exhilarating journey into the realm of Neural Networks, where bytes become brains and algorithms transform into genius thinkers. 🌐✨""")

st.write("""#### Create Your Digital Brain: Design Your Neural Network
Ready to play architect? Choose the number of convolutional layers, pick your favorite activation functions (ReLU, Sigmoid, or Tanh), and stack up linear layers to mold your digital masterpiece! 🛠️🎨""")

# Slider and multiselect for neural network design
num_conv_layers = st.selectbox("Number of Convolutional Layers", [1, 2,3])
activation_function = st.selectbox("Activation Functions", ["ReLU", "Sigmoid", "Tanh"])
num_linear_layers = st.selectbox("Number of Linear Layers", [1, 2, 3])
activation_function=activation_function.lower()
st.write("""#### Visualize the Magic: See Your Design Come Alive
Witness the magic unfold as your neural network design takes shape visually. Each layer represents a chapter in your network's story. Exciting, isn't it? 📊✨""")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation_function,
                 input_shape=(28,28,1)))
if num_conv_layers==2:
    model.add(Conv2D(128, (3, 3), activation=activation_function))
if num_conv_layers==3:
    model.add(Conv2D(128, (3, 3), activation=activation_function))
   # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation=activation_function))
model.add(Flatten())
if num_linear_layers==2:
    model.add(Dense(300, activation=activation_function))
   # model.add(keras.layers.Dropout(0.2))
if num_linear_layers==3:
    model.add(Dense(300, activation=activation_function))
    #model.add(keras.layers.Dropout(0.2))
    model.add(Dense(50, activation=activation_function))
    #model.add(keras.layers.Dropout(0.3))
model.add(Dense(10, activation='softmax'))
st.image(visualkeras.layered_view(model),use_column_width=True)

# Streamlit code for visualizing the designed neural network
# ...

# Section 2: Train Your Neural Network
st.write("""#### Ready, Set, Train: Unleash Your Neural Network!
Time to train your creation! Adjust the number of epochs and learning rate using sliders, as your network learns from the data and becomes a genius in its own right. ⏳🚂""")

# Sliders for training options
num_epochs = st.slider("Number of Epochs", 1, 11, step = 2)
learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)

st.subheader("Witness the Magic: Training Progress")
#@st.cache_resource(experimental_allow_widgets=True)
def train_nn(num_conv_layers, activation_function,num_linear_layers, num_epochs, learning_rate, _model):
    batch_size = 128
    num_classes = 10
    epochs = num_epochs
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
      #https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/ to know about image_data_format and what is "channelS_first"
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 #normalizing
    x_test /= 255 #normalizing

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    st.write('Test loss:', score[0])
    st.write('Test accuracy:', score[1]) 
    import pandas as pd
    df=pd.DataFrame(history.history)
    x = list(range(1,epochs+1))
    df["Epoch"]=x
    df=df[["Epoch","loss","accuracy","val_loss","val_accuracy"]]
    df=df.rename({"loss":"Training Loss","accuracy":"Training Accuracy","val_loss":"Test Loss","val_accuracy":"Test Accuracy"})
    #st.table(df)
    import matplotlib.pyplot as plt
    def plt_dynamic(x, vy, ty, ax, colors=['b']):
      ax.plot(x, vy, 'b', label="Validation Loss")
      ax.plot(x, ty, 'r', label="Train Loss")
      plt.legend()
      plt.grid()
      fig.canvas.draw()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
    # list of epoch numbers
    x = list(range(1,epochs+1))
    vy = history.history['val_loss']
    ty = history.history['loss']
    plt_dynamic(x, vy, ty, ax)
    #st.pyplot(fig)
    return model,fig, df
if st.button("Train Your Model"):
    st.write("Training In Progress")
    model, fig, df = train_nn(num_conv_layers, activation_function,num_linear_layers, num_epochs, learning_rate, model)
    if "model" not in st.session_state :
        st.session_state["model"]=model
    if "fig" not in st.session_state :
        st.session_state["fig"]=fig
    if "df" not in st.session_state :
        st.session_state["df"]=df
if "df" in st.session_state:
    st.table(st.session_state["df"])
if "fig" in st.session_state:   
    st.pyplot(st.session_state["fig"])

    
#model=train_nn(num_conv_layers, activation_function,num_linear_layers, num_epochs, learning_rate, model)

st.header("Evaluate Your Model")

st.markdown('''
Try to write a digit!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)
if "model" in st.session_state:
        model=st.session_state["model"]
SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28))
    st.write(f'Result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
    
user_feedback = st.radio("Is the prediction correct?", ("Yes", "No"))
if user_feedback == "Yes":
    st.success("Great! You have trained your model well.")
else:
    st.warning("Oh no! Improve upon hyperparameters to better tune your model.")
