import numpy as np
from PIL import Image
import os


def save_img(img: np.ndarray, dir: str, name: str) -> None:
    img = Image.fromarray((img*255.).clip(0,255).astype(np.uint8))
    img.save(os.path.join(dir, f'{name}.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # optimization conditions
    parser.add_argument('path', help='path to the input image file')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate')

    # size of MLP
    parser.add_argument('-w', '--width', type=int, default=32, help='hidden_width')
    parser.add_argument('-d', '--depth', type=int, default=2, help='hidden_depth')

    # encoding type
    parser.add_argument('--enc', default='none', help='type of input encoding')

    # Fourier feature encoding
    parser.add_argument('--freqs', type=int, default=10, help='number of fourier-feature frequencies')

    # hash encoding
    parser.add_argument('--min_res', type=int, default=16, help='minimum resolution of spatial grid')
    parser.add_argument('--levels', type=int, default=8, help='number of levels')
    parser.add_argument('--feature_dim', type=int, default=2, help='dimension of spatial features')
    parser.add_argument('--table_size_log2', type=int, default=14, help='table size in log2')

    # saving results
    parser.add_argument('--save_interval', type=int, default=4, help='interval between saving output images')
    parser.add_argument('--create_video', action='store_true', help='for creating a video of training progression')

    args = parser.parse_args()


    with Image.open(args.path) as img:
        img = (np.asarray(img).astype(np.float32)+.5)/256.
    

    OUTPUT_DIR = './output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    import eignn
    neural_field = eignn.NeuralField2D()

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
        print('falling back to no encoding')

        neural_field.set_network(hidden_dim, hidden_depth, out_dim)

    
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    img_dest = img.copy()
    epoch_id = 0
    interval = args.save_interval

    create_video = args.create_video
    if create_video:
        TMP_DIR = os.path.join(OUTPUT_DIR, 'tmp')
        os.makedirs(TMP_DIR, exist_ok=True)
        frame_id = 0

    
    while epoch_id < epochs:
        neural_field.fit(img, epoch_id, epoch_id+interval, epochs, batch_size, lr)
        neural_field.render(img_dest)

        if create_video:
            save_img(img_dest, TMP_DIR, f'{frame_id:03d}')
            frame_id += 1

        else:
            save_img(img_dest, OUTPUT_DIR, name='result')

        epoch_id += interval
 