from pathlib import Path

import cv2
import numpy as np
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text
from warnings import warn
from constants.enum_keys import PG
from pgdataset.s1_skeleton import PgdSkeleton
from aichallenger.s1_resize import ResizeKeepRatio
import pred.gesture_pred

class Player:
    def __init__(self, is_unittest=False):
        self.img_size = (512, 512)
        self.gpred = pred.gesture_pred.GesturePred()
        self.is_unittest = is_unittest
        self.target_fps = 15

    def play_dataset_video(self, is_train, video_index, show=True):
        self.gpred.reset()
        self.scd = PgdSkeleton(Path.home() / 'PoliceGestureLong', is_train, self.img_size)
        res = self.scd[video_index]
        print('Playing %s' % res[PG.VIDEO_NAME])
        coord_norm_FXJ = res[PG.COORD_NORM]  # Shape: F,X,J
        coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
        coord = coord_norm_FJX * np.array(self.img_size)
        img_shape = self.img_size[::-1] + (3,)
        kps = [KeypointsOnImage.from_xy_array(coord_JX, shape=img_shape) for coord_JX in coord]  # (frames, KeyOnImage)
        cap = cv2.VideoCapture(str(res[PG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/(v_fps*4))
        gestures = []  # Full video gesture recognition results
        for n in range(v_size):
            gdict = self.gpred.from_skeleton(coord_norm_FXJ[n][np.newaxis])
            gesture = gdict[PG.OUT_ARGMAX]
            gestures.append(gesture)
            if not show:
                continue
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            ges_name = self.gesture_dict[gesture]
            re_img = draw_text(re_img, 50, 100, ges_name, (255, 50, 50), size=40)
            pOnImg = kps[n]
            img_kps = pOnImg.draw_on_image(re_img)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", img_kps)
            key = cv2.waitKey(duration) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print('Stopped by user.')
                break
        cap.release()
        cv2.destroyAllWindows()
        gestures = np.array(gestures, np.int32)
        res[PG.PRED_GESTURES] = gestures
        print('The prediction of video ', res[PG.VIDEO_NAME], ' is completed')
        return res

    def play_custom_video(self, video_path):
        """video_path string: play video on disk
            video_path None: play video from camera on realtime
        """
        rkr = ResizeKeepRatio((512, 512))
        if video_path is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise IOError('Failed to open camera.')
            v_fps = self.target_fps
        else:
            cap = cv2.VideoCapture(str(video_path))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if v_fps != 15:
                warn('Suggested video frame rate is 15, currently %d. Frames will be sampled to 15 FPS.' % v_fps)
        duration = 10
        self.gpred.reset()
        frame_idx = 0
        next_sample = 0.0
        sample_step = max(float(v_fps or self.target_fps) / self.target_fps, 1.0)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if frame_idx + 0.5 < next_sample:
                frame_idx += 1
                continue
            next_sample += sample_step
            frame_idx += 1
            re_img, _, _ = rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            gdict = self.gpred.from_img(re_img)
            gesture = gdict[PG.OUT_ARGMAX]
            # Keypoints on image
            coord_norm_FXJ = gdict[PG.COORD_NORM]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            # Gesture name on image
            ges_name = self.gesture_dict[gesture]
            re_img = draw_text(re_img, 50, 100, ges_name, (255, 50, 50), size=40)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", re_img)
            key = cv2.waitKey(duration) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print('Stopped by user.')
                break
        cap.release()
        cv2.destroyAllWindows()


    gesture_dict = {
        0: "NO GESTURE",
        1: "STOP",
        2: "MOVE STRAIGHT",
        3: "LEFT TURN",
        4: "LEFT TURN WAITING",
        5: "RIGHT TURN",
        6: "LANG CHANGING",
        7: "SLOW DOWN",
        8: "PULL OVER"}

    gesture_dict_c = {
        0: "\u65e0\u624b\u52bf",
        1: "\u505c\u6b62",
        2: "\u76f4\u884c",
        3: "\u5de6\u8f6c",
        4: "\u5de6\u5f85\u8f6c",
        5: "\u53f3\u8f6c",
        6: "\u53d8\u9053",
        7: "\u51cf\u901f",
        8: "\u9760\u8fb9\u505c\u8f66"}
