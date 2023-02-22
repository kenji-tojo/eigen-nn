import os


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', help='path to the input directory')
    parser.add_argument('-f', '--framerate', type=int, default=30, help='framerate of an video')

    args = parser.parse_args()

    framerate = args.framerate
    dir = args.dir
    OUTPUT_DIR = './output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.system(f'ffmpeg -framerate {framerate} -i {dir}/%03d.png {OUTPUT_DIR}/video.mp4')