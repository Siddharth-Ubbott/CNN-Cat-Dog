from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(img_path):
    model = load_model('models/cat_dog_classifier.h5')
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Dog" if prediction > 0.5 else "Cat"

if __name__ == "__main__":
    img_path = 'dataset/test/cats/cat.4001.jpg'  # Example image
    print(f'The image is classified as: {predict_image(img_path)}')
