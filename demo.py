import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
with tf.device('/device:CPU:0'):
    model = tf.keras.models.load_model('keras_model2.h5', compile=False)
    LABELS = ['N', 'D']
    def predict(image):
    # Preprocess the image
        image = image.convert('RGB').resize((224, 224))
        image = np.array(image) / 255.0
        image = image.reshape(1, 224, 224, 3)
        prediction = model.predict(image)
        index = np.argmax(prediction)
        predicted_class = LABELS[index]
        confidence_score = prediction[0][index]

        return predicted_class, confidence_score

    def app():
        st.title('Deep Learning Model For D and N Class')
        file = st.file_uploader('Choose an image file')

        if file is not None:
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict(image)
            st.write('Prediction:', prediction)

    if __name__ == '__main__':
        app()
