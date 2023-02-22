import numpy as np
from PIL import Image
import os, shutil


OUTPUT_DIR = './output'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='path to the input image file')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate')

    parser.add_argument('-w', '--width', type=int, default=64, help='hidden_width')
    parser.add_argument('-d', '--depth', type=int, default=2, help='hidden_depth')

    # encoding type
    parser.add_argument('--enc', default='none', help='type of input encoding')

    # Fourier feature encoding
    parser.add_argument('-f', '--freqs', type=int, default=10, help='number of fourier-feature frequencies')

    # Hash encoding
    parser.add_argument('--min_res', type=int, default=16, help='base_res')
    parser.add_argument('--levels', type=int, default=3, help='levels')
    parser.add_argument('--feature_dim', type=int, default=2, help='feature_dim')
    parser.add_argument('--table_size_log2', type=int, default=14, help='table_size_log2')

    args = parser.parse_args()

    with Image.open(args.path) as img:
        img = (np.asarray(img).astype(np.float32)+.5)/256.

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import eignn

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    hidden_dim = args.width
    hidden_depth = args.depth

    if args.enc == 'none':
        print(f'encoding type: none')
        eignn.fit_field_vanilla(
            img, # will be updated in place
            epochs, batch_size, lr,
            hidden_dim, hidden_depth
        )
    elif args.enc == 'ff' or args.enc == 'fourier':
        print(f'encoding type: fourier feature')
        freqs = args.freqs
        eignn.fit_field_ff(
            img, # will be updated in place
            epochs, batch_size, lr,
            hidden_dim, hidden_depth,
            freqs
        )
    elif args.enc == 'hash':
        print(f'encoding type: hash')
        min_res = args.min_res
        feature_dim = args.feature_dim
        levels = args.levels
        table_size_log2 = args.table_size_log2
        eignn.fit_field_hash(
            img, # will be updated in place
            epochs, batch_size, lr,
            hidden_dim, hidden_depth,
            min_res, feature_dim, levels, table_size_log2
        )
    else:
        print(f'unknown encoding type {args.enc}')
        assert False

    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR,'result.png'))