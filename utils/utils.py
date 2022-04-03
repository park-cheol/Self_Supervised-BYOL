import torch
import torch.utils.data
import numpy as np


def init_target_network(online, target):
    """
    copy online network parameters and set requires_grad=False
    """
    for param_online, param_target in zip(online.parameters(), target.parameters()):
        param_target.data.copy_(param_online.data)
        param_target.requires_grad = False


@torch.no_grad()
def moving_average_update(online, target, momentum):
    """
    Update target network parameters
    """
    for param_online, param_target in zip(online.parameters(), target.parameters()):
        param_target.data = param_target.data * momentum + param_online.data * (1. - momentum)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def inference(loader, encoder, device):
    feature_vector = []
    labels_vector = []

    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = encoder(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size, args):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                              pin_memory=True, num_workers=args.workers)
    return train_loader, test_loader
