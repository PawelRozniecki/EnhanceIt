from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/dataset/T91'
TEST_DATAPATH = '/home/pawel/PycharmProjects/EnhanceIt/src/dataset/test'
BATCH_SIZE = 4
UPSCALE_FACTOR = 6
EPOCHS = 1500
SIZE = 32
MODEL_SAVE_PATH = '/home/pawel/PycharmProjects/EnhanceIt/srcbest.pth'
LEARNING_RATE = 10e-4
