"""NTHU-DDD four-state frame dataset (normal / drowsy / yawn / inattention).

Layout::

    <frames_root>/<class>/<image>.jpg   full frames sampled at 10 fps
    <faces_root>/<class>/<image>.jpg    MediaPipe FaceMesh face crops (same file name)

A frame without a face crop (face not detected) gets an all-zero face image.
Frame names keep the NTHU-DDD video name, the frame index and the four NTHU-DDD
annotations (fatigue, eye, head, mouth), e.g. ``001_glasses_sleepyCombination00610_1_1_0_0.jpg``.
"""
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = {"normal": 0, "drowsy": 1, "yawn": 2, "inattention": 3}
TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def read_list(path):
    """Lines of '<class>/<image>' in the order the model saw them."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class NTHUFrameDataset(Dataset):
    def __init__(self, items, frames_root, faces_root):
        self.items = list(items)
        self.frames_root, self.faces_root = frames_root, faces_root
        self.labels = [CLASSES[i.split("/")[0]] for i in self.items]

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        if os.path.isfile(path):
            return TRANSFORM(Image.open(path).convert("RGB"))
        return torch.zeros((3, 224, 224))

    def __getitem__(self, idx):
        item = self.items[idx]
        return (self._load(os.path.join(self.frames_root, item)),
                self._load(os.path.join(self.faces_root, item)),
                self.labels[idx])


def pad_collate(batch, batch_size):
    """Stack a batch; a short last batch is filled with random repeats of its own samples.

    The TCN runs along the batch, so every batch is kept at the same length.
    """
    frames, faces, labels = map(list, zip(*batch))
    if len(frames) < batch_size:
        extra = random.choices(range(len(frames)), k=batch_size - len(frames))
        frames += [frames[i] for i in extra]
        faces += [faces[i] for i in extra]
        labels += [labels[i] for i in extra]
    return torch.stack(frames), torch.stack(faces), torch.tensor(labels)
