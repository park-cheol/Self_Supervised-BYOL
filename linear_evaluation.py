import argparse
import random
import time
import datetime
import warnings
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from models.ResNet import ResNet
from models.logistic_regression import LogisticRegression
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--start-epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument('--data', default='./datasets', help='path to images')
parser.add_argument("--batch-size", type=int, default=256, help="size of the batches")
parser.add_argument("-j", "--workers", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

# Dataset Option
parser.add_argument('--image-size', default=96, help="Image Resolution")
# model option
parser.add_argument('--model-name', type=str, default='resnet50', choices=['resnet50', 'resnet18'], help='Backbone Model[Resnet50 or Resnet18]')
parser.add_argument('--hidden-channels', type=int, default=512, help="MLP hidden dimension")
parser.add_argument('--proj-channels', type=int, default=128, help="final Layer dimension")
parser.add_argument('--n-classes', type=int, default=10) # stl-10

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create encoder
    encoder = ResNet(model_name=args.model_name,
                     hidden_channels=args.hidden_channels,
                     proj_channels=args.proj_channels).cuda(args.gpu)
    classifier = LogisticRegression(encoder.projection.mlp[0].in_features,
                                    n_classes=args.n_classes).cuda(args.gpu)

    # load pretrained model
    if args.resume:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        encoder.load_state_dict(checkpoint['online'])
    else:
        raise Exception("Load Checkpoint ERROR")

    # remove projection
    encoder = nn.Sequential(*list(encoder.children())[:-1]).cuda(args.gpu)
    encoder.eval()
    print(encoder)
    # print("Check Encoder Requires_Grad")
    # for name, param in encoder.named_parameters():
    #     print(name, param.requires_grad)

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # cudnn.benchmark = True

    # DataSet & DataLoader
    data_transforms = transforms.Compose([transforms.Resize(size=args.image_size),
                                          transforms.ToTensor()])
    train_dataset = datasets.STL10(args.data, split='train', transform=data_transforms,
                                   download=True)
    test_dataset = datasets.STL10(args.data, split='test', transform=data_transforms,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers, pin_memory=True, drop_last=False)

    print("Representation from Pretrained BYOL(online network)")
    train_x, train_y = inference(train_loader, encoder, args.gpu)
    test_x, test_y = inference(test_loader, encoder, args.gpu)

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_x, train_y,
                                                                        test_x, test_y,
                                                                        args.batch_size,
                                                                        args)
    # Train
    for epoch in range(args.start_epoch, args.epochs):
        print("Training")
        train(arr_train_loader, classifier, criterion, optimizer, epoch, args, summary)
        print("Testing")
        acc1, acc5, loss = test(arr_test_loader, classifier, criterion, epoch, args, summary)
        print(" Epoch [%d] | loss: %f | Acc@1: %f | Acc@5: %f |"
              % (epoch + 1, loss, acc1, acc5))
        if acc1 >= best_acc1:
            print("[[Record Best Acc1]]")
            best_acc1 = acc1
    print("BEST ACC1 :", best_acc1)


def train(dataloader, classifier, criterion, optimizer, epoch, args, summary):
    classifier.train()

    start_time = time.time()
    for i, (image, target) in enumerate(dataloader):
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = classifier(image)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        niter = epoch * len(dataloader) + i
        summary.add_scalar('Linear_Eval/loss', loss.item(), niter)
        summary.add_scalar('Linear_Eval/acc1', acc1[0].item(), niter)
        summary.add_scalar('Linear_Eval/acc5', acc5[0].item(), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | loss: %f | Acc@1: %f | Acc@5: %f |"
                  % (epoch + 1, i, len(dataloader), loss, acc1[0], acc5[0]))

    elapse = datetime.timedelta(seconds=time.time() - start_time)
    print(f"걸린 시간: ", elapse)


def test(dataloader, classifier, criterion, epoch, args, summary):
    loss_avg = 0
    acc1_avg = 0
    acc5_avg = 0
    classifier.eval()

    for i, (image, target) in tqdm(enumerate(dataloader)):
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = classifier(image)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_avg += acc1[0]
        acc5_avg += acc5[0]
        loss_avg += loss

        niter = epoch * len(dataloader) + i
        summary.add_scalar('Lienar_Eval_test/loss', loss.item(), niter)
        summary.add_scalar('Lienar_Eval_test/acc1', acc1[0].item(), niter)
        summary.add_scalar('Lienar_Eval_test/acc5', acc5[0].item(), niter)

    summary.add_scalar('Lienar_Eval_test/avg_acc5', acc1_avg, epoch)
    summary.add_scalar('Lienar_Eval_test/avg_acc5', acc5_avg, epoch)

    return acc1_avg / len(dataloader), acc1_avg / len(dataloader), loss_avg / len(dataloader)


if __name__ == "__main__":
    main()