import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, RandomResizedCrop, RandomRotation
import copy
from collections import defaultdict


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.tensor(x.size(0), 1, 1, 1, dtype=torch.float32, device='cuda').bernoulli_(keep_prob)
    #mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):

  #if not os.path.exists(path):

  #remove old folder
  if os.path.exists(path):
    shutil.rmtree(path)
  os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  script_path = os.path.join(path, 'scripts')
  if not os.path.exists(script_path) and scripts_to_save is not None:
    os.mkdir(script_path)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def get_dataset(name, model_name=None, augmentation=False, resolution=32, val_split=0, balanced_val=True, autoaugment=True, cutout=True, cutout_length=16):

    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]

        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'flat_mnist':
        t = Compose([ToTensor(),
                     Normalize(
                         (0.1307,), (0.3081,)),
                     torch.nn.Flatten(0)
                     ])

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 28 * 28

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   Normalize((0.4376821, 0.4437697, 0.47280442),
                             (0.19803012, 0.20101562, 0.19703614))])

        t = [
            ToTensor(),
            Normalize((0.4376821, 0.4437697, 0.47280442),
                      (0.19803012, 0.20101562, 0.19703614))]

        # if 'resnet' in model_name:
        #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
        #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.SVHN(
            root='~/datasets/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/datasets/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]

        if resolution==32:
            # data processing used in NACHOS
            #tt = [Resize((resolution, resolution))]

            if augmentation:
                tt=[transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]

        else:
        
            tt = [RandomResizedCrop(resolution, scale=(0.08,1.0)),
                RandomHorizontalFlip()] #p=0.5 default]
        
        tt.extend([ ToTensor(),
                    Normalize(norm_mean, norm_std)
                    ])
                    
        
        '''
        tt = [RandomResizedCrop(resolution, scale=(0.08,1.0)),
                  #RandomCrop(32, padding=4),
                  RandomHorizontalFlip(), #p=0.5 default
                  #ToTensor(),
                  #Normalize(norm_mean, norm_std)
                  ]
        
        if autoaugment:
            tt.extend([CIFAR10Policy()])
        tt.extend([ToTensor()])
        if cutout:
            tt.extend([Cutout(cutout_length)])
        tt.extend([Normalize(norm_mean, norm_std)])
        '''

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize(norm_mean, norm_std)]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True,
            transform=transform)

        input_size, classes = (3, resolution, resolution), 10
        val_split=0.2

    elif name == 'cifar100':

        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([
                RandomCrop(resolution, padding=resolution//8),
                RandomHorizontalFlip(),
            ])

        tt.extend([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))])

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=False, download=True,
            transform=transform)

        input_size, classes = (3, resolution, resolution), 100
        val_split=0.2
    
    elif name == 'cinic10':

        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([RandomHorizontalFlip(),
                  RandomCrop(resolution, padding=resolution//8)])

        tt.extend([ToTensor(),
                   Normalize([0.47889522, 0.47227842, 0.43047404],
                             [0.24205776, 0.23828046, 0.25874835])])

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize([0.47889522, 0.47227842, 0.43047404],
                             [0.24205776, 0.23828046, 0.25874835])]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.ImageFolder('~/datasets/cinic10/train',
                                         transform=train_transform)
        test_set = datasets.ImageFolder('~/datasets/cinic10/test',
                                         transform=transform)

        input_size, classes = (3, resolution, resolution), 10

    elif name == 'tinyimagenet':
        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([
                RandomRotation(20),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
            ])
        else:
            tt.extend([
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
                ToTensor()])

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize((0.4802, 0.4481, 0.3975),
                      (0.2302, 0.2265, 0.2262))
        ]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.ImageFolder('../datasets/tiny-imagenet-200/train',
                                         transform=train_transform)
        test_set = datasets.ImageFolder('../datasets/tiny-imagenet-200/val',
                                         transform=transform)

        input_size, classes = (3, resolution, resolution), 200

    else:
        assert False

    val_set = None

    # Split the dataset into training and validation sets
    if val_split:
        train_len = len(train_set)
        eval_len = int(train_len * val_split)
        train_len = train_len - eval_len

        #print("VAL SPLIT: ", val_split)
        val_split=0.5

        if balanced_val:
            train_set, val_set = random_split_with_equal_per_class(train_set, val_split)

        else:
            train_set, val_set = torch.utils.data.random_split(train_set,
                                                        [train_len,
                                                            eval_len])

        val_set.dataset = copy.deepcopy(val_set.dataset)

        val_set.dataset.transform = test_set.transform
        val_set.dataset.target_transform = test_set.target_transform
        
    return train_set, val_set, test_set, input_size, classes

def random_split_with_equal_per_class(train_set, val_split):
    """
    Randomly shuffle and split a dataset into training and validation sets with an equal number of samples per class in the validation set.

    Args:
        train_set (Dataset): The dataset to split.
        val_split (float): The fraction of the dataset to include in the validation set.

    Returns:
        train_set (Subset): The training subset of the dataset.
        val_set (Subset): The validation subset of the dataset.
    """
    # Shuffle the train set
    train_size = len(train_set)
    shuffled_indices = torch.randperm(train_size).tolist()
    train_set = Subset(train_set, shuffled_indices)

    # Determine the number of samples per class for the validation set
    class_counts = defaultdict(int)
    for _, target in train_set:
        class_counts[target] += 1
    samples_per_class = {cls: int(val_split * count) for cls, count in class_counts.items()}

    #print("SAMPLES PER CLASS: ", samples_per_class)

    # Initialize lists to hold indices for the validation set
    val_indices = []

    # Iterate through the dataset to select samples for validation
    for cls in samples_per_class:
        class_indices = [idx for idx, (_, target) in enumerate(train_set) if target == cls]
        val_indices.extend(class_indices[:samples_per_class[cls]])

    # Create Subset with selected validation indices
    val_set = Subset(train_set, val_indices)

    # Remove the selected validation samples from the train_set
    train_indices = list(set(range(len(train_set))) - set(val_indices))
    train_set = Subset(train_set, train_indices)

    return train_set, val_set

def get_data_loaders(dataset, batch_size=32, threads=1, img_size=32, augmentation=False, val_split=0, eval_test=True):

    train_set, val_set, test_set,  _, _ = get_dataset(dataset, augmentation=augmentation, resolution=img_size, val_split=val_split)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads, pin_memory=True)

    if val_split:
        val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=threads, pin_memory=True)
    else:
        val_loader = None
    
    # Create DataLoader for test set if args.eval_test is True
    if eval_test:
        test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=threads, pin_memory=True)
    else:
        test_loader=None
    
    return train_loader, val_loader, test_loader

