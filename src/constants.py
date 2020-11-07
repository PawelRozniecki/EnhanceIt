from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/dataset/T91/1'
TEST_DATAPATH = '/home/pawel/PycharmProjects/EnhanceIt/src/dataset/T91/test'
BATCH_SIZE = 2
UPSCALE_FACTOR = 2
EPOCHS = 150
SIZE = 32
MODEL_SAVE_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/models/generator.pth'
DIS_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/models/discriminatorModel.pth'
LEARNING_RATE = 10e-5
