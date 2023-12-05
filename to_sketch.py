from sketchKeras_pytorch.src.model import SketchKeras
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm

folder_path = "./dataset/archive/dataset/full_colour"
output_path = "./dataset/archive/dataset/sketch"
weights_path = "./sketchKeras_pytorch/weights/model.pth"


# note to get the model head over to https://github.com/higumax/sketchKeras-pytorch.git
device = "cuda" if torch.cuda.is_available() else "mps"


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 3), dtype=np.float)
    ret[0:h, 0:w, 0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":
    model = SketchKeras().to(device)

    model.load_state_dict(torch.load(weights_path))

    for img_path in tqdm(os.listdir(folder_path)):
        # check if the sketch already exists
        if os.path.exists(os.path.join(output_path, img_path)):
            print(f"{img_path} already exists")
            continue

        join_path = os.path.join(folder_path, img_path)
        img = cv2.imread(join_path)
        if img is None:
            print(f"could not read {join_path}")
            continue

        # resize
        height, width = float(img.shape[0]), float(img.shape[1])
        if width > height:
            new_width, new_height = (512, int(512 / width * height))
        else:
            new_width, new_height = (int(512 / height * width), 512)
        img = cv2.resize(img, (new_width, new_height))

        # preprocess
        img = preprocess(img)
        x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
        x = torch.tensor(x).float()

        # feed into the network
        with torch.no_grad():
            pred = model(x.to(device))
        pred = pred.squeeze()

        # postprocess
        output = pred.cpu().detach().numpy()
        output = postprocess(output, thresh=0.1, smooth=False)
        output = output[:new_height, :new_width]

        final_output_path = os.path.join(output_path, img_path)

        cv2.imwrite(final_output_path, output)
        print(f"saved {final_output_path}")
