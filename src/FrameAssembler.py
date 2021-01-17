import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import os
import cv2


import argparse

# parser = argparse.ArgumentParser(description='Enter file path')
# parser.add_argument('--path', type=str, required=True, help='path to the file you want to assemble')

def assemble(path):
    # args = parser.parse_args()

    try:

        os.chdir(path)
        os.system('ffmpeg -r 25 -f image2 -s 2560x1440 -i frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p .mp4 output.mp4')
    except OSError as err:
        print(err)


