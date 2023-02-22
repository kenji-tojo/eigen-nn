import numpy as np
from PIL import Image
import os, shutil


def save_img(img: np.ndarray, dir: str, name: str) -> None:
    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(dir, f'{name}.png'))


OUTPUT_DIR = './output'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # optimization conditions
    parser.add_argument('path', help='path to the input image file')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--interval', type=int, default=4, help='interval to save image')

    # size of MLP
    parser.add_argument('-w', '--width', type=int, default=64, help='hidden_width')
    parser.add_argument('-d', '--depth', type=int, default=2, help='hidden_depth')

    # encoding type
    parser.add_argument('--enc', default='none', help='type of input encoding')

    # Fourier feature encoding
    parser.add_argument('-f', '--freqs', type=int, default=10, help='number of fourier-feature frequencies')

    # hash encoding
    parser.add_argument('--min_res', type=int, default=16, help='base_res')
    parser.add_argument('--levels', type=int, default=3, help='levels')
    parser.add_argument('--feature_dim', type=int, default=2, help='feature_dim')
    parser.add_argument('--table_size_log2', type=int, default=14, help='table_size_log2')

    # saving animation
    parser.add_argument('--create_video', action='store_true', help='create video of training progression')
    parser.add_argument('--framerate', type=int, default=30, help='framerate of animation')

    args = parser.parse_args()

    with Image.open(args.path) as img:
        img = (np.asarray(img).astype(np.float32)+.5)/256.

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import eignn

    neural_field = eignn.NeuralField2D()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    hidden_dim = args.width
    hidden_depth = args.depth
    out_dim = img.shape[2]

    if args.enc == 'none' or args.enc == '':
        print(f'encoding type: none')

        neural_field.set_network(hidden_dim, hidden_depth, out_dim)

    elif args.enc == 'ff' or args.enc == 'fourier':
        print(f'encoding type: fourier feature')

        freqs = args.freqs

        neural_field.set_network_ff(hidden_dim, hidden_depth, out_dim, freqs)

    elif args.enc == 'hash':
        print(f'encoding type: hash')

        min_res = args.min_res
        feature_dim = args.feature_dim
        levels = args.levels
        table_size_log2 = args.table_size_log2

        neural_field.set_network_hash(
            hidden_dim, hidden_depth, out_dim,
            min_res, feature_dim, levels, table_size_log2
        )
    
    else:
        print(f'unknown encoding type {args.enc}')
        assert False

    
    img_dest = img.copy()
    epoch_id = 0
    interval = args.interval

    create_video = args.create_video
    frame_id = 0
    if create_video:
        TMP_DIR = os.path.join(OUTPUT_DIR, 'tmp')
        os.makedirs(TMP_DIR, exist_ok=True)
    
    while epoch_id < epochs:
        neural_field.fit(img, epoch_id, epoch_id+interval, epochs, batch_size, lr)
        neural_field.render(img_dest)

        if create_video:
            save_img(img_dest, TMP_DIR, f'{frame_id:03d}')

        else:
            save_img(img_dest, OUTPUT_DIR, name='result')

        epoch_id += interval
        frame_id += 1

    
    if create_video:
        framerate = args.framerate
        os.system(f'ffmpeg -framerate {framerate} -i {TMP_DIR}/%03d.png {OUTPUT_DIR}/video.mp4')
        # shutil.rmtree(TMP_DIR)
 