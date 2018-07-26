from config import *
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mvcnn_model import MVConv
from dataset import ModelNetDataset, ToTensor
from torch.utils.data import Dataset, DataLoader
from utils import accuracy, AverageMeter


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()
print_freq=10




def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to validation mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input= input.cuda()


        in1 = input[:,0,:,:,:]
        in2 = input[:,1,:,:,:]
        in3 = input[:,2,:,:,:]

        if use_gpu:
           input_var1 = torch.autograd.Variable(in1.float().cuda())
           input_var2 = torch.autograd.Variable(in2.float().cuda())
           input_var3 = torch.autograd.Variable(in3.float().cuda())
           target_var = torch.autograd.Variable(target.long().cuda())

        else:
           input_var1 = torch.autograd.Variable(in1.float())
           input_var2 = torch.autograd.Variable(in2.float())
           input_var3 = torch.autograd.Variable(in3.float())
           target_var = torch.autograd.Variable(target.long())


        # compute output
        output = model(input_var1,input_var2,input_var3)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg




if __name__ == '__main__':
    # Load the trained weights of multi-view VGG16_bn CNN model
    if len(sys.argv) > 1:
        trained_model = sys.argv[1]
    else: trained_model = trained_model_path_best

    model = MVConv()
    model.load_state_dict(torch.load(trained_model))

    if use_gpu:
        model.cuda()

    print(model)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    # Loading the test data
    transformed_valid_dataset = ModelNetDataset(root_dir=train_data_root,
                                                phase='test',
                                                transform=transforms.Compose([
                                                    ToTensor()
                                                ]))

    # Loading dataset into dataloader
    test_loader = torch.utils.data.DataLoader(transformed_valid_dataset, batch_size=test_batch_size,
                                              shuffle=True, num_workers=num_workers)

    prec1 = test(test_loader, model, criterion)
