import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import os
import cv2


def assemble():
    try:
        os.system('ffmpeg -r 60 -f image2 -s 320x240 -i frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4')
    except OSError as err:
        print(err)


def main():
    assemble()


main()
