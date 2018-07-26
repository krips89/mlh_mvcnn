## general settings.

#path to folder containing MLH features
train_data_root = '/sarkar/ML/ML_MN_int_256_5l_3v/'

#path to the saved model
trained_model_path = './data/vgg16_current_model.pth.tar'

#default path to the best model
trained_model_path_best = './data/vgg16_bn_best_model.pth.tar'
trained_model_path_paper = './data/vgg16_bn_best_model.pth.tar'


#training parameters.
lr = 0.01                                                                       # learning rate
lr_decay = 0.1                                                                  # adjust_learning_rate
epochs = 20                                                                     # total number of epochs
weight_decay = 1e-5                                                             # L2 regularization
train_batch_size = 8                                                                  # set batch size
test_batch_size = 4
num_workers = 8                                                                 # number of workers
num_classes = 40                                                                # number of classes
