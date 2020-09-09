import bz2
from faceswap.face_alignment import image_align

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_image(img, landmark_detector):
    face_landmarks = landmark_detector.get_landmarks(img)
    return image_align(img, face_landmarks)

