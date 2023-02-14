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
        img = np.asarray(img).astype(np.float32)/255.


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import eignn
    img = eignn.fit_nn(img)
    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR, 'result.png'))