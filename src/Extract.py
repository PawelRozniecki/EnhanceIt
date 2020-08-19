import cv2
import sys
import os
from tqdm import tqdm


def extractAudio(path):
    print("extracting audio in progress")

    try:
        file, extension = os.path.splitext(path)
        os.system('ffmpeg -i {file}{ext} {file}.wav'.format(file=file, ext=extension))
        os.system('lame {file}.wav {file}.mp3'.format(file=file))
        os.remove('{}.wav'.format(file))

        print('"{}" successfully converted into MP3'.format(path))

    except OSError as err:
        print(err)
        exit(-1)


def extract():
    path = sys.argv[1]

    if path is not None:
        extractAudio(path)
        video_capture = cv2.VideoCapture(path)
        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total)
        success, image = video_capture.read()
        count = 0
        while success:
            for count in tqdm(range(total)):
                cv2.imwrite("/home/pawel/PycharmProjects/EnhanceIt/src/images/frame%d.jpg" % count, image)  # save frame as jpeg  file
                success, image = video_capture.read()
                # print("new frame: ", success)
                count += 1
    else:
        print("provide a path to a file")

    print("extracted all the frames successfully")

def main():
    extract()


main()