from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/Users/pingwin/PycharmProjects/EnhanceIt/src/dataset/T91/1'
TEST_DATAPATH = '/Users/pingwin/PycharmProjects/EnhanceIt/src/dataset/T91/test'
BATCH_SIZE = 7
UPSCALE_FACTOR = 4
EPOCHS = 20
SIZE = 88
MODEL_SAVE_PATH = '/Users/pingwin/PycharmProjects/EnhanceIt/src/models/bestSRGAN.pth'
LEARNING_RATE = 10e-5