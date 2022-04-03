import argparse
import os
import random
import time
import datetime
import warnings

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from data.transform import *
from models.ResNet import ResNet
from models.MLP import MLP
from utils.losses import Regression_loss
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--start-epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument('--data', default='./datasets', help='path to images')
parser.add_argument("--batch-size", type=int, default=256, help="size of the batches")
parser.add_argument("-j", "--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

# Dataset Option
parser.add_argument('--image-size', default=96, help="Image Resolution")
# model option
parser.add_argument('--model-name', type=str, default='resnet50', choices=['resnet50', 'resnet18'], help='Backbone Model[Resnet50 or Resnet18]')
parser.add_argument('--hidden-channels', type=int, default=512, help="MLP hidden dimension")
parser.add_argument('--proj-channels', type=int, default=128, help="final Layer dimension")
parser.add_argument('--momentum-update', type=float, default=0.996, help="slow moving average update for target parameters")

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create models
    online_network = ResNet(model_name=args.model_name,
                            hidden_channels=args.hidden_channels,
                            proj_channels=args.proj_channels)
    predictor = MLP(in_channels=online_network.projection.mlp[-1].out_features,
                    hidden_channels=args.hidden_channels,
                    proj_channels=args.proj_channels)

    target_network = ResNet(model_name=args.model_name,
                            hidden_channels=args.hidden_channels,
                            proj_channels=args.proj_channels)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            online_network.cuda(args.gpu)
            predictor.cuda(args.gpu)
            target_network.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            online_network = torch.nn.parallel.DistributedDataParallel(online_network, device_ids=[args.gpu])
            predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[args.gpu])
            target_network = torch.nn.parallel.DistributedDataParallel(target_network, device_ids=[args.gpu])
        else:
            online_network.cuda()
            predictor.cuda()
            target_network.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            online_network = torch.nn.parallel.DistributedDataParallel(online_network)
            predictor = torch.nn.parallel.DistributedDataParallel(predictor)
            target_network = torch.nn.parallel.DistributedDataParallel(target_network)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        online_network = online_network.cuda(args.gpu)
        predictor = predictor.cuda(args.gpu)
        target_network = target_network.cuda(args.gpu)
    else:
        online_network = torch.nn.DataParallel(online_network).cuda()
        predictor = torch.nn.DataParallel(predictor).cuda()
        target_network = torch.nn.DataParallel(target_network).cuda()

    # Loss & Optimizer
    criterion = Regression_loss
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                lr=0.03,
                                momentum=0.9,
                                weight_decay=0.0004
                                )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            online_network.load_state_dict(checkpoint['online'])
            predictor.load_state_dict(checkpoint['predictor'])
            target_network.load_state_dict(checkpoint['target'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    simclr_transform = get_simclr_data_transforms(input_shape=(args.image_size, args.image_size, 3))
    train_dataset = datasets.STL10(root=args.data, split='train+unlabeled', download=True,
                                   transform=MultiViewDataInjector([simclr_transform, simclr_transform]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # Train
    init_target_network(online=online_network,
                        target=target_network)
    print("check target network Requires_Grad")
    for name, param in target_network.named_parameters():
        if not param.requires_grad: # All False
            continue
        else:
            print(name, param.requires_grad)
            raise Exception("requires_grad error")

    print("check target network initialize")
    for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
        if torch.all((param_q.data == param_k.data)):
            continue
        else:
            raise Exception("copy error")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, online_network, target_network, predictor,
              criterion, optimizer, epoch, summary, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save({
                'epoch': epoch + 1,
                'online': online_network.state_dict(),
                'target': target_network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(train_loader, online_network, target_network, predictor,
          criterion, optimizer, epoch, summary, args):

    end = time.time()
    for i, ((img_1, img_2), _) in enumerate(train_loader):
        # print(type(img_1))
        img_1 = img_1.cuda(args.gpu, non_blocking=True)
        img_2 = img_2.cuda(args.gpu, non_blocking=True)

        predict_1 = predictor(online_network(img_1))
        predict_2 = predictor(online_network(img_2))

        with torch.no_grad():
            target_1 = target_network(img_1).detach().clone()
            target_2 = target_network(img_2).detach().clone()

        loss_1 = criterion(predict_1, target_1)
        loss_2 = criterion(predict_2, target_2)
        # print(loss_1)
        # print(loss_2)
        loss = (loss_1 + loss_2).mean()
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        moving_average_update(online=online_network,
                              target=target_network,
                              momentum=args.momentum_update)

        niter = epoch * len(train_loader) + i
        summary.add_scalar('Train/Loss', loss.item(), niter)
        # summary.add_scalar('Train/Loss_1', loss_1, niter)
        # summary.add_scalar('Train/Loss_2', loss_2, niter)

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch+1}][{i}/{len(train_loader)}] | Loss: {loss: .4f} |")

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


if __name__ == "__main__":
    main()
