
import numpy as np
import os
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split


def _get_normalization(net_name: str):
    """Pretrained-weight-appropriate (mean, std). ArcFace IResNet was trained
    with (x - 127.5) / 128, i.e. mean=0.5/std=0.5 after ToTensor → [0,1].
    Everything else uses ImageNet stats."""
    if net_name.startswith('iresnet'):
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# Number of landmarks per file (MediaPipe Face Mesh 468 + 10 iris = 478).
NUM_LANDMARKS = 478

# Extra landmark directories always searched as a fallback, in order, after the
# caller's landmarks_dir. The CASIA-WebFace landmarks hold the normal/control faces
# used as the negative class in the binary task (their ids are absent from the GMDB
# landmark folder), so resolution must look here too.
EXTRA_LANDMARK_DIRS = [
    '/data2/masino_lab/abamini/facial_data/casia_webface_align_landmarks/',
]


def landmark_path_for_image(image_path: str, landmarks_dir) -> str:
    """Map an image path to its landmark file, trying the known naming conventions.

    GMDB aligned faces use "<id>_aligned_aligned_landmarks.txt" (image "<id>.jpg" or
    "<id>_aligned.jpg"); CASIA-WebFace aligned faces use "<stem>_aligned_landmarks.txt"
    (image "<stem>_aligned.jpg"). `landmarks_dir` may be a single directory or a list;
    EXTRA_LANDMARK_DIRS (e.g. the CASIA normal-face landmarks) are searched after it.
    Returns the first candidate that exists on disk, else the GMDB-style path in the
    primary dir (so a genuinely missing file is still reported as missing)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = stem[:-len('_aligned')] if stem.endswith('_aligned') else stem
    candidates = [
        base + '_aligned_landmarks.txt',  # GMDB
        stem + '_landmarks.txt',                   # CASIA (stem already ends with _aligned)
        stem + '_aligned_landmarks.txt',           # CASIA when stem lacks _aligned
    ]
    dirs = [landmarks_dir] if isinstance(landmarks_dir, str) else list(landmarks_dir)
    dirs += [d for d in EXTRA_LANDMARK_DIRS if d not in dirs]
    for d in dirs:
        for name in candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return os.path.join(dirs[0], candidates[0])


def load_landmarks(path: str, img_size: int, num_landmarks: int = NUM_LANDMARKS):
    """Parse a "<id>,<x>,<y>" landmark file into a (num_landmarks, 2) array of
    normalized (row=y, col=x) coordinates in [0,1]. Missing entries / file are -1."""
    arr = np.full((num_landmarks, 2), -1.0, dtype=np.float32)
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                idx = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                if 0 <= idx < num_landmarks:
                    arr[idx, 0] = y / img_size  # row
                    arr[idx, 1] = x / img_size  # col
    except FileNotFoundError:
        pass
    return arr


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset =='CUB-200-2011':     
        return get_birds(True, '/data2/masino_lab/abamini/Datasets/CUB_200_2011/dataset/train_crop', '/data2/masino_lab/abamini/Datasets/CUB_200_2011/dataset/train', '/data2/masino_lab/abamini/Datasets/CUB_200_2011/dataset/test_crop', args.image_size, args.seed, args.validation_size, '/data2/masino_lab/abamini/Datasets/CUB_200_2011/dataset/train', '/data2/masino_lab/abamini/Datasets/CUB_200_2011/dataset/test_full')
    if args.dataset == 'pets':
        return get_pets(True, '/data2/masino_lab/abamini/Datasets/PETS/dataset/train','/data2/masino_lab/abamini/Datasets/PETS/dataset/train','/data2/masino_lab/abamini/Datasets/PETS/dataset/test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'partimagenet': #use --validation_size of 0.2
        return get_partimagenet(True, '/data2/masino_lab/abamini/Datasets/partimagenet/dataset/all', '/data2/masino_lab/abamini/Datasets/partimagenet/dataset/all', None, args.image_size, args.seed, args.validation_size) 
    if args.dataset == 'CARS':
        return get_cars(True, '/data2/masino_lab/abamini/Datasets/CARS/dataset/train', '/data2/masino_lab/abamini/Datasets/CARS/dataset/train', '/data2/masino_lab/abamini/Datasets/CARS/dataset/test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'grayscale_example':
        return get_grayscale(True, '/data2/masino_lab/abamini/Datasets/grayscale_example/dataset/train', '/data2/masino_lab/abamini/Datasets/grayscale_example/dataset/train', '/data2/masino_lab/abamini/Datasets/grayscale_example/dataset/test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'gmdb':
        pretrain_dir = args.pretrain_data_dir if args.pretrain_data_dir != '' else None
        # Region-masked pooling (--masked_pool) needs per-image landmarks and the
        # geometry-preserving augmentation that keeps them valid for the augmented views.
        need_landmarks = getattr(args, 'masked_pool', False)
        return get_gmdb(True, '/data2/masino_lab/abamini/facial_data/Final_data/gmdb_train', '/data2/masino_lab/abamini/facial_data/Final_data/gmdb_train', '/data2/masino_lab/abamini/facial_data/Final_data/gmdb_val', args.image_size, args.seed, args.validation_size, train_dir_pretrain=pretrain_dir, net_name=args.net, need_landmarks=need_landmarks, landmarks_dir=getattr(args, 'landmarks_dir', None), landmark_img_size=getattr(args, 'landmark_img_size', 112), pretrain_landmarks_dir=getattr(args, 'pretrain_landmarks_dir', None))
    if args.dataset == 'gmdb_binary':
        # Binary task: a 2-class ImageFolder (positive syndromic vs normal/control faces).
        # TODO: set these two paths to your binary ImageFolder dirs. Each must contain
        # exactly two class subfolders; name the POSITIVE class so it sorts FIRST
        # alphabetically (the binary metrics treat class index 0 as positive).
        # Normal-face landmarks are resolved from CASIA via EXTRA_LANDMARK_DIRS.
        binary_train_dir = '/data2/masino_lab/abamini/facial_data/combined_normal_and_rare/train_crop/'  
        binary_test_dir  = '/data2/masino_lab/abamini/facial_data/combined_normal_and_rare/validation_crop/'    
        pretrain_dir = args.pretrain_data_dir if args.pretrain_data_dir != '' else None
        need_landmarks = getattr(args, 'masked_pool', False)
        return get_gmdb(True, binary_train_dir, binary_train_dir, binary_test_dir, args.image_size, args.seed, args.validation_size, train_dir_pretrain=pretrain_dir, net_name=args.net, need_landmarks=need_landmarks, landmarks_dir=getattr(args, 'landmarks_dir', None), landmark_img_size=getattr(args, 'landmark_img_size', 112), pretrain_landmarks_dir=getattr(args, 'pretrain_landmarks_dir', None))

def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    
    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain


    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
                                        
    else:        
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes

def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None, landmarks_dir=None, landmark_img_size=112, pretrain_landmarks_dir=None):

    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices,test_size=validation_size,stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which",len(train_indices),"for training and ", len(test_indices),"for testing.", flush=True)
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    
    if landmarks_dir is not None:
        trainset = torch.utils.data.Subset(LandmarkAugDataset(trainvalset, transform1=transform1, transform2=transform2, landmarks_dir=landmarks_dir, landmark_img_size=landmark_img_size), indices=train_indices)
    else:
        trainset = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        has_subdirs = any(os.path.isdir(os.path.join(train_dir_pretrain, e)) for e in os.listdir(train_dir_pretrain))
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain) if has_subdirs else FlatImageDataset(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            stratify_pr = subset_targets_pr if len(set(subset_targets_pr)) > 1 else None
            train_indices_pr, test_indices_pr = train_test_split(indices_pr,test_size=validation_size,stratify=stratify_pr, random_state=seed)

        if pretrain_landmarks_dir is not None:
            # Supply landmarks for region-masked pooling already during pretraining.
            # transform1 is geometry-preserving when landmarks are used (just a Resize),
            # so the precomputed landmarks stay valid for both augmented views.
            pretrain_base = LandmarkAugDataset(trainvalset_pr, transform1=transform1p or transform1, transform2=transform2, landmarks_dir=pretrain_landmarks_dir, landmark_img_size=landmark_img_size)
        else:
            pretrain_base = TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p or transform1, transform2=transform2)
        trainset_pretraining = torch.utils.data.Subset(pretrain_base, indices=train_indices_pr)
    else:
        trainset_pretraining = None
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

def get_pets(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        
        transform2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)), #includes crop
        transforms.ToTensor(),
        normalize
        ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_partimagenet(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, test_dir_projection, transform1p)

def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
       
        transform2 = transforms.Compose([
                    TrivialAugmentWideNoShapeWithColor(),
                    transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                    transforms.ToTensor(),
                    normalize
                    ])
                            
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_grayscale(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.Grayscale(3), #convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_gmdb(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain=None, net_name: str = 'resnet', need_landmarks=False, landmarks_dir=None, landmark_img_size=112, pretrain_landmarks_dir=None):
    mean, std = _get_normalization(net_name)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment and need_landmarks:
        # Landmark-safe augmentation: keep geometry fixed (deterministic resize, no flip/crop)
        # so the precomputed landmark coordinates remain valid for both augmented views.
        # Only photometric augmentation differs between the two views.
        transform1 = transforms.Resize(size=(img_size, img_size))
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.ToTensor(),
            normalize
        ])
    elif augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+16, img_size+16)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, landmarks_dir=(landmarks_dir if need_landmarks else None), landmark_img_size=landmark_img_size, pretrain_landmarks_dir=(pretrain_landmarks_dir if need_landmarks else None))

class FlatImageDataset(torch.utils.data.Dataset):
    """Loads images from a flat directory (no class subfolders) with dummy targets for self-supervised pretraining."""
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __init__(self, root):
        from PIL import Image as PILImage
        self._pil_image = PILImage
        self.root = root
        self.classes = ['face']
        self.class_to_idx = {'face': 0}
        self.imgs = [
            (os.path.join(root, f), 0)
            for f in sorted(os.listdir(root))
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ]
        self.targets = [0] * len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self._pil_image.open(path).convert('RGB')
        return img, target

    def __len__(self):
        return len(self.imgs)


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if hasattr(dataset, 'imgs') and hasattr(dataset, 'targets'):
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


class LandmarkAugDataset(torch.utils.data.Dataset):
    r"""Like TwoAugSupervisedDataset, but also returns per-image MediaPipe landmarks.

    Returns (view1, view2, target, landmarks) where landmarks has shape
    (NUM_LANDMARKS, 2) with normalized (row=y, col=x) coordinates in [0,1]
    (-1 marks a missing landmark/file). Landmarks are precomputed once at init
    in the order of the underlying ImageFolder samples.
    """
    def __init__(self, dataset, transform1, transform2, landmarks_dir, landmark_img_size=112, num_landmarks=NUM_LANDMARKS):
        self.dataset = dataset
        self.classes = dataset.classes
        self.imgs = dataset.imgs
        self.targets = dataset.targets
        self.transform1 = transform1
        self.transform2 = transform2

        lms = []
        missing = 0
        for path, _ in self.imgs:
            lm_path = landmark_path_for_image(path, landmarks_dir)
            arr = load_landmarks(lm_path, landmark_img_size, num_landmarks)
            if (arr < 0).all():
                missing += 1
            lms.append(arr)
        self.landmarks = torch.from_numpy(np.stack(lms, axis=0))  # (N, L, 2)
        print("LandmarkAugDataset: loaded landmarks for %d/%d images (missing %d) from %s" % (
            len(lms) - missing, len(lms), missing, landmarks_dir), flush=True)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target, self.landmarks[index]

    def __len__(self):
        return len(self.dataset)

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

