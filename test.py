import numpy as np
from PIL import Image
import os, shutil


OUTPUT_DIR = './output'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to the input image file')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('-f', '--freqs', type=int, default=10, help='number of frequencies')
    parser.add_argument('-s', '--step_size', type=float, default=1e0, help='step size of gradient descent')
    parser.add_argument('-w', '--width', type=int, default=64, help='hidden_width')
    parser.add_argument('-d', '--depth', type=int, default=2, help='hidden_depth')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--min_res', type=int, default=16, help='base_res')
    parser.add_argument('--levels', type=int, default=3, help='levels')
    parser.add_argument('--feature_dim', type=int, default=2, help='feature_dim')
    parser.add_argument('--table_size_log2', type=int, default=14, help='table_size_log2')
    args = parser.parse_args()

    with Image.open(args.path) as img:
        img = (np.asarray(img).astype(np.float32)+.5)/256.


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import eignn

    hidden_dim = args.width
    hidden_depth = args.depth
    step_size = args.step_size
    batch_size = args.batch_size
    epochs = args.epochs
    freqs = args.freqs
    min_res = args.min_res
    levels = args.levels
    feature_dim = args.feature_dim
    table_size_log2 = args.table_size_log2
    eignn.fit_nn(
        img, # will be updated in place
        hidden_dim, hidden_depth,
        step_size, batch_size, epochs,
        min_res, levels, feature_dim, table_size_log2,
        np.array([2**f for f in range(freqs)], dtype=np.float32)
    )
    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR,'result.png'))