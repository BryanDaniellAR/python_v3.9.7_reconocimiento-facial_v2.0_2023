import cv2
import os
import imutils
import numpy as np

def apply_color_effect(frame, color_effect):
    if color_effect == 'normal':
        return frame
    elif color_effect == 'gray':
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    elif color_effect == 'dark_gray':
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_gray_frame = cv2.addWeighted(gray_frame, 0.7, np.zeros_like(gray_frame), 0.3, 0)
        return cv2.cvtColor(dark_gray_frame, cv2.COLOR_GRAY2BGR)
    else:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        light_gray_frame = cv2.addWeighted(gray_frame, 0.3, np.ones_like(gray_frame) * 255, 0.7, 0)
        return cv2.cvtColor(light_gray_frame, cv2.COLOR_GRAY2BGR)

def capture_images(person_name, total_images, images_per_color):
    data_path = 'Fotos'
    person_path = os.path.join(data_path, person_name)

    if not os.path.exists(person_path):
        print('Carpeta creada:', person_path)
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    color_range = ['normal', 'gray', 'dark_gray', 'light_gray']

    count = 0
    while count < total_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        aux_frame = frame.copy()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = aux_frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)

            color_effect = color_range[count // images_per_color % len(color_range)]
            color_effect_frame = apply_color_effect(face_roi, color_effect)
            cv2.imwrite(os.path.join(person_path, f'rostro_{count}_{color_effect}.jpg'), color_effect_frame)
            count += 1

        cv2.imshow('Captura', frame)
        k = cv2.waitKey(1)
        if k == 27 or count >= total_images:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    person_name = input("El nombre de la persona capturada es:")
    total_images = 300
    images_per_color = 50
    capture_images(person_name, total_images, images_per_color)