import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image


def extract_amp_spectrum(trg_img):
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target


def amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    return a_local


def freq_space_interpolation(local_img, amp_target, L=0, ratio=0):
    local_img_np = local_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.xticks([])
    plt.yticks([])

    return 0


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def freq_space_interpolation_batch(buffer_img_batch, current_img_batch, L=0.003, ratio=0):
    assert buffer_img_batch.shape == current_img_batch.shape

    # Randomly choose a ratio from [0.2, 0.4, 0.6, 0.8, 1.0]
    ratio = random.choice([0.2, 0.4, 0.6, 0.8, 1.0])

    # Randomly choose a target image from current batch
    target_img_idx = random.randint(0, current_img_batch.shape[0]-1)
    target_img = current_img_batch[target_img_idx]

    # Extract the amplitude spectrum of the target image
    amp_target = extract_amp_spectrum(target_img)

    # Perform frequency space interpolation on each image in the buffer
    buffer_in_current = []
    for img in buffer_img_batch:
        img_in_current = freq_space_interpolation(img, amp_target, L=L, ratio=1-ratio)
        buffer_in_current.append(img_in_current)
    buffer_in_current_tensor = [torch.from_numpy(array) for array in buffer_in_current]

    return torch.stack(buffer_in_current_tensor, axis=0)

# Set the random seed
random.seed(42)