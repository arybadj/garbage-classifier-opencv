import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('garbage_classifier.h5')

# Get class labels (run this once to get labels, or hardcode from validation_generator)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'trashnet/data/dataset-resized',  # Update path if different locally
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    subset='validation'
)
class_labels = list(validation_generator.class_indices.keys())

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    predicted_label = class_labels[predicted_class]

    # Display the result
    label = f"{predicted_label}: {confidence:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classifier', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()