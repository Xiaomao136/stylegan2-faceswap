import dlib
import PIL
import cv2
class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, img):
        det_rets = self.detector(img, 1)
        for detection in det_rets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks

    def get_landmarks_from_file(self, filename):
        img = dlib.load_rgb_image(filename)
        self.get_landmarks(img)
