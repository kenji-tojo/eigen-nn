import numpy as np
from PIL import Image
import os, shutil


OUTPUT_DIR = './output'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to the input image file')
    args = parser.parse_args()

    with Image.open(args.path) as img:
        img = (np.asarray(img).astype(np.float32)+.5)/256.


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import eignn
    hidden_dim = 32
    hidden_depth = 1
    step_size = 1e0
    batch_size = 128
    epochs = 20
    img = eignn.fit_nn(img, hidden_dim, hidden_depth, step_size, batch_size, epochs)
    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR,'result.png'))