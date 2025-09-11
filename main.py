from rec.face_det_image import  ImageFaceRecognizer
import time

from rec.live_face_detection import RealTimeFaceRecognizer

if __name__ == '__main__':
    # image_path = "C:\\Users\\techma\Downloads\\test_img1.jpg"  # Change to your image
    recognizer = RealTimeFaceRecognizer(

        providers=['CPUExecutionProvider'],  # Use CPUExecutionProvider if no GPU
        use_gpu=True
    )
    recognizer.run()


