# app.py

import streamlit as st

def start_your_journey_section():
    st.header("🚀 Start Your Journey")

    st.write("Welcome to the MNIST Exploration Room! Here, you'll embark on a journey to understand the magic behind machine learning models.")
    
    st.write("##### 🎩 Introduction to Machine Learning Models:")
    st.write("Machine Learning is like giving computers a superpower - the ability to learn from data and make predictions. Imagine your computer distinguishing between cats and dogs on its own !")
    st.image("pages/how computers see.png",use_column_width=True)
    
    st.write("##### 🕵️ How do Machine Learning Algorithms Work?")
    st.write("Think of these algorithms as intelligent detectives. They learn patterns from examples, making decisions and predictions based on what they've seen before. It's like having a virtual detective solving puzzles for you! ")
    st.write("##### 🌟 What can Machine Learning Algorithms do?")
    st.write("From recognizing handwritten digits to powering virtual assistants, machine learning is everywhere! It's the force behind personalized recommendations, image recognition, and so much more. Get ready to be amazed by the possibilities! ")
    #st.write("##### 🎯 Your Quest Begins:")
    #st.write("From recognizing handwritten digits to powering virtual assistants, machine learning is everywhere! It's the force behind personalized recommendations, image recognition, and so much more. Get ready to be amazed by the possibilities! ")
    
    
#    st.write("##### Your Quest Begins:")
#    st.write("Click the button below to start your journey and explore the fascinating world of machine learning!")

 #   if st.button("Start Your Journey"):
 #       st.markdown("<a href='#' target='_blank'>Let the Adventure Begin...</a>", unsafe_allow_html=True)  # Add relevant URL for the actual content

def explore_the_dataset_section():
    st.header("🔍 Explore the Dataset")

    st.write("Get ready to dive into the MNIST dataset, a famous collection of handwritten digits!")

    st.write("##### Introduction to the MNIST Dataset:")
    st.write("The MNIST dataset is a classic in the ML community, consisting of 28x28 pixel grayscale images of handwritten digits (0-9). This dataset serves as an excellent starting point for understanding image classification.")

    st.markdown( """**Quick Facts About MNIST** """)
    st.write("- 70,000 images depicting handwritten digits")
    st.write("- Each image is labeled with the corresponding digit")
    st.write("- Pixel values represent the darkness of the pixel (0 to 255)")
    st.write("- Grayscale images, so only one channel")

    st.write("##### Your Mission:")
    st.write("Ready to get hands-on experience with the dataset. Head to the Explore the Dataset section to interact with samples from the dataset. Understanding your dataset is the first step towards building a successful model. The more you understand your data, the more informed decisions you can make when training your model.")

    if st.button("Explore the Dataset"):
        st.markdown("<a href='ExploreTheDataset' target='_blank'>Dive into the Digits...</a>", unsafe_allow_html=True)  # Add relevant URL for the actual content

def algorithm_selection_section():
  

    st.header("🌐Hands-On Experience ")
    st.write("🧠 Craft your own spells by connecting magical neurons. Learn the basics of spellcasting and see how your apprentice interprets the runes! ")
    st.write("A deep learning algorithm with multiple layers, suitable for various tasks, including image classification.")

    #st.write("##### Convolutional Neural Networks (CNNs):")
    #st.write("🎨 Dive deep into the realm of deep learning with convolutional magic. Uncover the secrets of magical filters and their role in decoding handwritten runes! ")
    #st.write("A specialized deep learning algorithm for image-related tasks, leveraging convolutional layers to extract hierarchical features.")


    st.write("##### Ready to Experiment?")
    st.write("Tap the button below and start your hands-on exploration!")

  #  algorithms = ["KNN", "Neural Networks"]  # Add more algorithms as needed
  #  selected_algorithm = st.selectbox("Select Algorithm", algorithms)

    if st.button(f"Start Neural Networks Adventure"):
        st.markdown(f"<a href='DesignYourOwnNetwork' target='_blank'>Redirecting...</a>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Exploration Room",
        page_icon="🔬",
        layout="centered",
    )

    # Header Section
    st.title("MNIST Exploration Room: Uncover the Magic of Machine Learning")
    
    # Start Your Journey Section
    start_your_journey_section()

    # Explore the Dataset Section
    explore_the_dataset_section()

    # Algorithm Selection Section
    algorithm_selection_section()

if __name__ == "__main__":
    main()
