# app.py

import streamlit as st

def main():
    st.set_page_config(
        page_title="MNIST Playground",
        page_icon="🤖",
        layout="centered",
    )
    
    # Header Section
    st.title("MNIST Playground: Learn Machine Learning the Fun Way!")
    st.write("No Code, Just Play! Dive into the World of MNIST with Interactive Challenges.")

    # Hero Section
    st.image("landing page.jpeg",use_column_width=True)

    # Introduction Section
    st.write("Welcome to MNIST Playground, where you'll embark on an exciting machine learning journey without writing a single line of code!")
    st.write("MNIST, a dataset of handwritten digits, is the perfect starting point for beginners. It's fun, interactive, and a great way to understand the basics of machine learning.")
    st.write("No code, no lectures. Just exploration, challenges, and a lot of fun!")

    # CTA Button
    if st.button("Explore MNIST Now"):
        st.markdown("<a href='ExplorationRoom' target='_blank'>Redirecting...</a>", unsafe_allow_html=True)

    # Footer Section
    st.markdown("---")
    st.write("© 2023 MNIST Playground. All Rights Reserved.")
    st.write("Privacy Policy | Terms of Service | Contact Us")

    # Optional: Newsletter Subscription
    st.write("Stay Updated!")
    email = st.text_input("Enter your email:")
    if st.button("Subscribe"):
        st.write(f"Subscribed with {email}!")  # Add the subscription logic here

if __name__ == "__main__":
    main()

