import pickle
import numpy as np
import cv2 as cv
import pretrained_networks
from keras.utils import get_file
import bz2
import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
import dlib
from faceswap.landmarks_detector import LandmarksDetector
from faceswap.face_alignment import image_align
from faceswap.dataset import create_from_images
from faceswap.projector import project_real_images

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


class FaceSwapper:
    # FaceSwapper初始化
    def __init__(self, network_pkl, video_file, style_image_file, col_styles):
        self.video_file = video_file
        self.style_image_file = style_image_file
        self.col_styles = col_styles

        # 加载stylegan2人脸模型
        if network_pkl is None:
            self._G = self._D = self.Gs = None
        elif not network_pkl.startswith('gdrive:'):
            with open(network_pkl, "rb") as f:
                self.pf = f
                self._G, self._D, self.Gs = pickle.load(f)
        else:
            self._G, self._D, self.Gs = pretrained_networks.load_networks(network_pkl)
        # 加载landmark模型
        landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                   LANDMARKS_MODEL_URL, cache_subdir='temp'))
        self.landmarks_detector = LandmarksDetector(landmarks_model_path)
        self.style_latent_vector = None

    # 图像对齐（提取脸部，并裁剪成1024*1024的大小）
    def _align_image(self, img):
        # process faces
        for i, face_landmarks in enumerate(self.landmarks_detector.get_landmarks(np.array(img)), start=1):
            if face_landmarks is None:
                continue
            return image_align(img, face_landmarks)

    # 图像投影到潜在空间，并获取潜码
    def _project(self, image, dataset_dir, seq_no, style=False):
        dataset_name = 'content'
        snapshot_name = 'tmp/content_snapshots'
        if style:
            dataset_name = 'style'
            snapshot_name = 'tmp/style_snapshots'

        img = np.asarray(image)
        tfrecord_dir = dataset_dir + dataset_name
        if not dataset_dir.endswith('/'):
            tfrecord_dir = dataset_dir + '/' + dataset_name

        # 把图像转换为tfrecrod格式
        create_from_images(img, tfrecord_dir)
        # get latent
        return project_real_images(self.Gs, dataset_dir, dataset_name, snapshot_name, seq_no)

    def _styleMixing(self, content_latent_vector, style_latent_vector, col_styles):
        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = 4
        Gs_syn_kwargs.truncation_psi = 1.0
        w = style_latent_vector[0].copy()
        w[col_styles] = content_latent_vector[0][col_styles]
        image = self.Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
        return image

    # 从风格图像中获取投影到潜在空间的latent
    def _get_style_latent(self, seq_no):
        img = PIL.Image.open(self.style_image_file)
        # 图像对齐
        aligned_img = self._align_image(img)
        # 投影到潜在空间，并获取latent
        return self._project(aligned_img, 'tmp/datasets', seq_no, style=True)

    # 从内容图像中获取投影在潜在空间的latent
    def _get_content_latent(self, img, seq_no):
        image = PIL.Image.fromarray(img)
        # 图像对齐
        aligned_img = self._align_image(image)
        # 投影到潜在空间，并获取latent
        return self._project(aligned_img, 'tmp/datasets', seq_no)

    def _mixing(self, content_latent, style_latent, col_styles):
        z_content = np.stack(content_latent[0] for _ in range(1))
        z_style = np.stack(style_latent[0] for _ in range(1))
        w_content = self.Gs.components.mapping.run(z_content, None)
        w_style = self.Gs.components.mapping.run(z_style, None)
        w_content[0] = content_latent
        w_style[0] = style_latent

        return self._styleMixing(w_content, w_style, col_styles)

    # 把风格混合的脸，替换到视频帧图想中
    def _swap(self, img, mixed_face):
        video_frame = PIL.Image.fromarray(img)
        video_frame.save('tmp/video_frame.png')
        mixed_face.save('tmp/mixed_face.png')
        return True

    # 给视频文件进行换脸操作
    def face_swap(self):
        # set noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        # process style
        self.style_latent_vector = self._get_style_latent(0)
        np.save(dnnlib.make_run_dir_path('style.npy'), self.style_latent_vector)

        # prcess video
        cap = cv.VideoCapture(self.video_file)
        seq_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exting ...")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 获取latent
            latent = self._get_content_latent(img, seq_no)
            np.save(dnnlib.make_run_dir_path('content_%d.npy' % seq_no, latent))
            # 风格迁移混合
            mixed_face = self._mixing(latent, self.style_latent_vector, self.col_styles)
            # 换脸
            self._swap(img, mixed_face)
            seq_no += 1
            break

    # 测试
    def test(self):
        style_latent = np.load('results/src1/me_01.npy')
        content_latent = np.load('results/dst/100-100_01.npy')
        mixed_face = self.mixing(content_latent, style_latent, [0, 1, 2, 3, 4, 5, 6])
        mixed_face.show()