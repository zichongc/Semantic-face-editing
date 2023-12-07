"""
Face align the in-the-wild CelebA, obtain the aligned faces with size of 224x224.
Download and place `shape_predictor_68_face_landmarks.dat` to `./checkpoints` for preparation.
"""

from PIL import Image
import dlib
import numpy as np
from torchvision import transforms
import scipy
import cv2.cv2 as cv

import os
import argparse
from tqdm import tqdm


def get_landmark(filepath, predictor_):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    assert len(dets) > 0, "Face not detected, try another face image"

    shape = predictor_(img, dets[0])

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm, dets[0]


def align_face(filepath, output_size=1024, return_tensor=True):
    lm, rect = get_landmark(filepath, predictor)
    l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()

    img = cv.imread(filepath)
    cv.rectangle(img, (l, t), (r, b), color=(0, 0, 255))

    for index, pt in enumerate(lm):
        pt_pos = tuple(pt)
        cv.circle(img, pt_pos, 2, (0, 0, 255), 2)

    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)  # normalize
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.open(filepath)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    # print(border)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))

    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')[:, :, :3]
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))

        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return transforms.ToTensor()(img) if return_tensor else img


if __name__ == '__main__':
    predictor = dlib.shape_predictor("../../checkpoints/shape_predictor_68_face_landmarks.dat")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./images')
    parser.add_argument('--output', type=str, default='./images_aligned')

    args = parser.parse_args()
    files = os.listdir(args.dataset)

    dataset = args.dataset
    output = args.output
    os.makedirs(output, exist_ok=True)

    failed_list = []
    for file in tqdm(files):
        file_path = os.path.join(dataset, file)
        output_path = os.path.join(output, file)

        try:
            face = align_face(file_path, output_size=224, return_tensor=False)
            face.save(output_path)
        except AssertionError:
            failed_list.append(file)

    with open('./filtered.txt', 'w') as file:
        for i in failed_list:
            file.write(i+'\n')
