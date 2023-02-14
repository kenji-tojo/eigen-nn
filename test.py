import numpy as np
from PIL import Image
import os, shutil


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='path to the input file')
    args = parser.parse_args()

    if args.file:
        print(f'input file: {args.file}')
    

    import eignn

    mat = np.zeros((10,20), dtype=np.float32)
    eignn.inspect_tensor(mat)