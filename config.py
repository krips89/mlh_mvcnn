### general settings ###

#path to folder containing MLH features
train_data_root = '/sarkar/ML/ML_MN_int_256_5l_3v/'

#path to the saved model
trained_model_path = './data/vgg16_current_model.pth.tar'

#default path to the best model
trained_model_path_best = './data/vgg16_bn_best_model.pth.tar'
trained_model_path_paper = './data/ vgg16_bn_paper_model.pth.tar'


#training parameters.
lr = 0.01
lr_decay = 0.1
epochs = 20
weight_decay = 1e-5
train_batch_size = 4
test_batch_size = 4
num_workers = 8
num_classes = 40
