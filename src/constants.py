from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/Datasets/DIV2K/Train'
TEST_DATAPATH = '/home/pawel/PycharmProjects/EnhanceIt/src/Datasets/DIV2K/Validation'
BATCH_SIZE = 3
UPSCALE_FACTOR = 2
EPOCHS = 100
SIZE = 96
MODEL_SAVE_PATH = '/src/models/SRGAN_Trained.pth'
DIS_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/dis.pth'
LEARNING_RATE = 10e-5
CHECKPOINT_DIR ='/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints'
CP  = '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/testcp'