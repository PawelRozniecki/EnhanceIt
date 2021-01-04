import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import cv2
import sys
import os
from src.data_utils import extract_filename
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.constants import ENHANCED_FRAMES_DIR
from src.constants import EXTRACTED_FRAMES_DIR


class FrameExtractor:

    def __init__(self, path):
        self.path = path

    def extract(self):
        # path = sys.argv[1]

        dirname = extract_filename(self.path)
        print(dirname)
        current_path = ENHANCED_FRAMES_DIR + dirname
        print(current_path)
        try:
            os.mkdir(current_path)
        except OSError :
            print("Creation of path %s failed" % current_path)
        else:
            print("Path %s created" % current_path)

        if self.path is not None:
            # extractAudio(path)
            video_capture = cv2.VideoCapture(self.path)
            total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print(total)
            success, image = video_capture.read()
            count = 0
            while success:
                for count in tqdm(range(total)):
                    cv2.imwrite("/home/pawel/PycharmProjects/EnhanceIt/src/extracted_frames/frame%d.png" % count,
                                image)  # save frame as jpeg  file
                    success, image = video_capture.read()
                    # print("new frame: ", success)
                    count += 1
        else:
            print("provide a path to a file")

        print("extracted all the frames successfully")
        self.set_path(current_path)

    def get_path(self):
        return self.path + '/'

    def set_path(self, path):
        self.path = path




# def extractAudio(path):
#     print("extracting audio in progress")
#
#     try:
#         file, extension = os.path.splitext(path)
#         os.system('ffmpeg -i {file}{ext} {file}.wav'.format(file=file, ext=extension))
#         os.system('lame {filpythoe}.wav {file}.mp3'.format(file=file))
#         os.remove('{}.wav'.format(file))
#
#         print('"{}" successfully converted into MP3'.format(path))
#
#     except OSError as err:
#         print(err)
#         exit(-1)


#
# def main():
#      extract()
#
#
# main()
