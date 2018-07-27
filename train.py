import shutil
from config import *
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
from utils import accuracy, AverageMeter
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def save_checkpoint(state_dict, is_best, filename='./data/vgg16_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, trained_model_path_best)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    l_rate = lr * (lr_decay ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = l_rate

use_gpu = torch.cuda.is_available()
print_freq=10

def main():

    best_prec1 = 0
    model = MVConv()
    if use_gpu:
        model.cuda()
    print(model)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    print(lr,weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                weight_decay = weight_decay)

    cudnn.benchmark = True

    transformed_train_dataset = ModelNetDataset(root_dir=train_data_root,
                                           phase='train',
                                           transform=transforms.Compose([
                                           ToTensor()
                                           ]))

    transformed_valid_dataset = ModelNetDataset(root_dir=train_data_root,
                                           phase='test',
                                           transform=transforms.Compose([
                                           ToTensor()
                                           ]))

    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(transformed_train_dataset, batch_size=train_batch_size,
                                               shuffle=True, num_workers=num_workers)

    val_loader =  torch.utils.data.DataLoader(transformed_valid_dataset, batch_size=test_batch_size,
                                               shuffle=True, num_workers=num_workers)

    start_time= time.time()

    # Train for all epochs between Startepoch, Endepoch
    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train on train dataset
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model.state_dict(), is_best, trained_model_path)

    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        in1 = input[:,0,:,:,:]
        in2 = input[:,1,:,:,:]
        in3 = input[:,2,:,:,:]

        if use_gpu:
           in1 = torch.autograd.Variable(in1.float().cuda())
           in2 = torch.autograd.Variable(in2.float().cuda())
           in3 = torch.autograd.Variable(in3.float().cuda())
           target_var = torch.autograd.Variable(target.long().cuda())

        else:
           in1 = torch.autograd.Variable(in1.float())
           in2 = torch.autograd.Variable(in2.float())
           in3 = torch.autograd.Variable(in3.float())
           target_var = torch.autograd.Variable(target.long())


        # compute output
        output = model(in1,in2,in3)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to validation mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
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
   main()