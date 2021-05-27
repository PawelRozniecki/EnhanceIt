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
        os.system('e')
    except OSError as err:
        print(err)


