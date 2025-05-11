import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist_cnn_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(frame_gray, (28, 28))
    roi = roi.astype('float32') / 255.0
    roi = roi.reshape(1, 28, 28, 1)

    pred = model.predict(roi)
    label = np.argmax(pred)

    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Digit Classification", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()