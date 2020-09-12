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
        self.network_pkl = network_pkl

        # 加载stylegan2人脸模型
        self._load_network_pkl()
        # 加载landmark模型
        # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
        #                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))
        landmarks_model_path = unpack_bz2('model/shape_predictor_68_face_landmarks.dat.bz2')
        self.landmarks_detector = LandmarksDetector(landmarks_model_path)
        self.style_latent_vector = None

    # 加载stylegan2人脸模型
    def _load_network_pkl(self):
        if self.network_pkl is None:
            self._G = self._D = self.Gs = None
        elif not self.network_pkl.startswith('gdrive:'):
            with open(self.network_pkl, "rb") as f:
                self.pf = f
                self._G, self._D, self.Gs = pickle.load(f)
        else:
            self._G, self._D, self.Gs = pretrained_networks.load_networks(self.network_pkl)

    # 图像对齐（提取脸部，并裁剪成1024*1024的大小）
    def _align_image(self, img):
        # process faces
        for i, face_landmarks in enumerate(self.landmarks_detector.get_landmarks(np.array(img)), start=1):
            return image_align(img, face_landmarks)
        return None, None

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
        aligned_img, crop = self._align_image(img)
        # 投影到潜在空间，并获取latent
        return self._project(aligned_img, 'tmp/datasets', seq_no, style=True)

    # 从内容图像中获取投影在潜在空间的latent
    def _get_content_latent(self, img, seq_no):
        image = PIL.Image.fromarray(img)
        # 图像对齐
        aligned_img, crop = self._align_image(image)
        if aligned_img is None or crop is None:
            return None, None
        # 投影到潜在空间，并获取latent
        return self._project(aligned_img, 'tmp/datasets', seq_no), crop

    def _mixing(self, content_latent, style_latent, col_styles):
        z_content = np.stack(content_latent[0] for _ in range(1))
        z_style = np.stack(style_latent[0] for _ in range(1))
        w_content = self.Gs.components.mapping.run(z_content, None)
        w_style = self.Gs.components.mapping.run(z_style, None)
        w_content[0] = content_latent
        w_style[0] = style_latent

        return self._styleMixing(w_content, w_style, col_styles)

    # 把风格混合的脸，替换到视频帧图想中
    def _swap(self, img, face_mixed, crop):
        image = PIL.Image.fromarray(img)
        mixed = PIL.Image.fromarray(face_mixed)
        face_resized = mixed.resize((crop[2] - crop[0], crop[3] - crop[1]), PIL.Image.ANTIALIAS)
        image.paste(face_resized, (crop[0], crop[1]))
        return np.array(image)

    # 给视频文件进行换脸操作
    def face_swap(self, start):
        if self.Gs is None:
            print('Gs network is none')
            exit(1)

        # set noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

        # process style
        # self.style_latent_vector = self._get_style_latent(0)
        # np.save(dnnlib.make_run_dir_path('npy/style.npy'), self.style_latent_vector)
        self.style_latent_vector = np.load('npy/style.npy')

        # process video
        cap = cv.VideoCapture(self.video_file)
        seq_no = 0
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv.CAP_PROP_FPS)
        size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        splits = self.video_file.split('.')
        result_video_file = splits[0] + '_result.' + splits[1]
        fourcc = cv.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv.VideoWriter(result_video_file, fourcc, fps, size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exting ...")
                break
            if seq_no >= start:
                img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # 获取latent
                latent, crop = self._get_content_latent(img, seq_no)
                if latent is not None and crop is not None:
                    content_npy_file = 'npy/content_{}.npy'.format(seq_no)
                    np.save(dnnlib.make_run_dir_path(content_npy_file), latent)
                    # 风格迁移混合
                    face_mixed = self._mixing(latent, self.style_latent_vector, self.col_styles)
                    mixed_file = 'mixed/{}.png'.format(seq_no)
                    PIL.Image.fromarray(face_mixed).save(mixed_file)
                    # 换脸
                    face_swapped = self._swap(img, face_mixed, crop)
                    swapped_file = 'swapped/{}.png'.format(seq_no)
                    PIL.Image.fromarray(face_swapped).save(swapped_file)
                    # write to video
                    face_bgr = cv.cvtColor(face_swapped, cv.COLOR_RGB2BGR)
                    video_writer.write(face_bgr)
                else:
                    video_writer.write(frame)
            else:
                swapped_file = 'swapped/{}.png'.format(seq_no)
                face_swapped = PIL.Image.open(swapped_file)
                # write to video
                face_bgr = cv.cvtColor(np.array(face_swapped), cv.COLOR_RGB2BGR)
                video_writer.write(face_bgr)

            # increment seq_no
            seq_no += 1
            print('process {}/{} ......'.format(seq_no, frames))
            if seq_no > 100:
                break
        video_writer.release()

    # test1
    def test1(self):
        if self.Gs is None:
            print('Gs network is none')
            exit(1)

        # set noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        # process style
        self.style_latent_vector = np.load('npy/style.npy')

        # process video
        cap = cv.VideoCapture(self.video_file)
        seq_no = 0
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv.CAP_PROP_FPS)
        size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        splits = self.video_file.split('.')
        result_video_file = splits[0] + '_result.' + splits[1]
        fourcc = cv.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv.VideoWriter(result_video_file, fourcc, fps, size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exting ...")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 获取latent
            latent = np.load('npy/content_0.npy')
            if latent is not None:
                content_npy_file = 'npy/content_{}.npy'.format(seq_no)
                np.save(dnnlib.make_run_dir_path(content_npy_file), latent)
                # 风格迁移混合
                face_mixed = self._mixing(latent, self.style_latent_vector, self.col_styles)
                mixed_file = 'mixed/{}.png'.format(seq_no)
                PIL.Image.fromarray(face_mixed).save(mixed_file)
            else:
                video_writer.write(frame)
            seq_no += 1
            print('process {}/{} ......'.format(seq_no, frames))
            break
        video_writer.release()

    # 测试图像合并
    def test2(self):
        # prcess video
        cap = cv.VideoCapture(self.video_file)
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv.CAP_PROP_FPS)
        size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        splits = self.video_file.split('.')
        result_video_file = splits[0] + '_result.' + splits[1]
        fourcc = cv.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv.VideoWriter(result_video_file, fourcc, fps, size)
        seq_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exting ...")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            image = PIL.Image.fromarray(img)
            #image.show()
            # 图像对齐
            aligned_img, crop = self._align_image(image)
            if aligned_img is not None and crop is not None:
                resize_img = aligned_img.resize((crop[2] - crop[0], crop[3] - crop[1]), PIL.Image.ANTIALIAS)
                image.paste(resize_img, (crop[0], crop[1]))
                face_bgr = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                video_writer.write(face_bgr)
            else:
                video_writer.write(frame)
            seq_no += 1
            print('process {}/{} ......'.format(seq_no, frames))
        video_writer.release()

    def test3(self, start):
        if self.Gs is None:
            print('Gs network is none')
            exit(1)

        # set noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

        # process style
        # self.style_latent_vector = self._get_style_latent(0)
        # np.save(dnnlib.make_run_dir_path('npy/style.npy'), self.style_latent_vector)
        self.style_latent_vector = np.load('npy/style.npy')

        # process video
        cap = cv.VideoCapture(self.video_file)
        seq_no = 0
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exting ...")
                break

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 获取latent
            latent, crop = self._get_content_latent(img, seq_no)
            if latent is not None and crop is not None:
                content_npy_file = 'npy/content_{}.npy'.format(seq_no)
                np.save(dnnlib.make_run_dir_path(content_npy_file), latent)
                # 风格迁移混合
                face_mixed = self._mixing(latent, self.style_latent_vector, self.col_styles)
                mixed_file = 'mixed/{}.png'.format(seq_no)
                PIL.Image.fromarray(face_mixed).save(mixed_file)

            # increment seq_no
            seq_no += 1
            print('process {}/{} ......'.format(seq_no, frames))