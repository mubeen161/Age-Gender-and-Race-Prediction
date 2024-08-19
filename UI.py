import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
gender_dict = {0: "Male", 1: "Female"}
race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224)) 
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  
    return img

def make_prediction():
    global img_label

    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    age, gender, race = predictions[0][0], np.argmax(predictions[1]), np.argmax(predictions[2])
    result_text.set(f"Age: {int(age)}\nGender: {gender_dict[gender]}\nRace: {race_dict[race]}")
    img = Image.open(file_path)
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    img_label.configure(image=img)
    img_label.image = img
root = tk.Tk()
root.title("Image Prediction")
img_label = Label(root)
img_label.pack()
result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Arial", 16))
result_label.pack()
predict_button = Button(root, text="Load Image and Predict", command=make_prediction)
predict_button.pack()
root.mainloop()
