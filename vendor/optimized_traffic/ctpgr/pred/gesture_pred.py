from collections import deque
from typing import Iterable
from constants.enum_keys import HK, PG
from models.gesture_recognition_model import GestureRecognitionModel
from models.pose_estimation_model import PoseEstimationModel
import torch
import numpy as np

from pgdataset.s3_handcraft import BoneLengthAngle
from pred.human_keypoint_pred import HumanKeypointPredict


class TemporalGestureSmoother:
    def __init__(self, window_size=7, min_stable_frames=3, confidence_floor=0.35):
        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.confidence_floor = confidence_floor
        self.labels = deque(maxlen=window_size)
        self.probs = deque(maxlen=window_size)
        self.last_label = None

    def reset(self):
        self.labels.clear()
        self.probs.clear()
        self.last_label = None

    def update(self, logits):
        probs = self._softmax(logits)
        raw_label = int(np.argmax(probs))
        self.labels.append(raw_label)
        self.probs.append(probs)

        avg_probs = np.mean(np.asarray(self.probs), axis=0)
        candidate = int(np.argmax(avg_probs))
        candidate_votes = sum(1 for label in self.labels if label == candidate)

        if self.last_label is not None:
            stable_enough = candidate_votes >= self.min_stable_frames
            confident_enough = avg_probs[candidate] >= self.confidence_floor
            if not stable_enough and not confident_enough:
                candidate = self.last_label

        self.last_label = candidate
        return candidate, avg_probs

    @staticmethod
    def _softmax(logits):
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)


class GesturePred:
    def __init__(self, smoothing=True, smooth_window=7, min_stable_frames=3, confidence_floor=0.35):
        self.p_predictor = HumanKeypointPredict()
        self.bla = BoneLengthAngle()
        self.g_model = GestureRecognitionModel(1)
        self.g_model.load_ckpt(allow_new=False)
        self.g_model.eval()
        self.smoothing = smoothing
        self.smoother = TemporalGestureSmoother(
            window_size=smooth_window,
            min_stable_frames=min_stable_frames,
            confidence_floor=confidence_floor)
        self.reset()

    def reset(self):
        self.h, self.c = self.g_model.h0(), self.g_model.c0()
        self.smoother.reset()

    def from_skeleton(self, coord_norm):
        # coord_norm: FXJ, F==1
        assert coord_norm.ndim == 3 and coord_norm.shape[0] == 1
        ges_data = self.bla.handcrafted_features(coord_norm)  # Shape: (F, C) F==1
        features = np.concatenate((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                              ges_data[PG.BONE_ANGLE_SIN]), axis=1)
        features = features[np.newaxis]
        features = features.transpose((1, 0, 2))  # NFC->FNC
        features = torch.from_numpy(features)
        features = features.to(self.g_model.device, dtype=torch.float32)
        with torch.no_grad():
            _, h, c, class_out = self.g_model(features, self.h, self.c)  # Output shape: (1, 1, num_classes)
        self.h, self.c = h.detach(), c.detach()
        np_out = class_out[0].detach().cpu().numpy()
        if self.smoothing:
            max_arg, np_out = self.smoother.update(np_out)
        else:
            np_out = TemporalGestureSmoother._softmax(np_out)
            max_arg = int(np.argmax(np_out))
        res_dict = {PG.OUT_ARGMAX: max_arg, PG.OUT_SCORES: np_out, PG.COORD_NORM: coord_norm}
        return res_dict

    def from_img(self, img: np.ndarray):

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8 and img.ndim == 3, "Expect ndarray of shape (H, W, C)"
        p_res = self.p_predictor.get_coordinates(img)
        res_dict = self.from_skeleton(p_res[PG.COORD_NORM][np.newaxis])
        return res_dict
