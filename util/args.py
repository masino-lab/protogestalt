import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
from datetime import datetime

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PIP-Net')
    parser.add_argument('--dataset',
                        type=str,
                        default='gmdb',
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. iresnet100 is the ArcFace face-recognition backbone (expects 112x112 input, produces a 14x14 prototype grid via the layer4 stride=1 modification). Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26, convnext_tiny_13, iresnet100.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=80,
                        help='The number of epochs PIP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default = 5,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.02, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0003, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0001, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-3,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--run_number',
                        type=str,
                        default=None,
                        help='Run identifier. Defaults to current datetime (YYYYMMDD_HHMMSS) if not set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='/data2/masino_lab/abamini/pipnet/runs/run_pipnet',
                        help='Base directory for logging. Run number will be appended automatically')
    parser.add_argument('--num_features',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--image_size',
                        type=int,
                        default=112,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument('--pretrain_data_dir',
                        type=str,
                        default='',
                        help='Directory of a separate dataset (ImageFolder structure) to use for prototype pretraining (Stage 1). If empty, the main training dataset is used for pretraining.')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 14,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    parser.add_argument('--face_landmarker_model',
                        type=str,
                        default='/data2/masino_lab/abamini/models/face_landmarker.task',
                        help='Path to the MediaPipe FaceLandmarker .task model file. Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    parser.add_argument('--extra_test_image_folder',
                        type=str,
                        default='./experiments',
                        help='Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments')
    parser.add_argument('--no_vis',
                        action='store_true',
                        help='Skip all visualization and prototype image saving. Speeds up tuning runs significantly — only accuracy metrics are logged.')
    parser.add_argument('--unfreeze_backbone_pretrain',
                        action='store_true',
                        help='Train the whole backbone (incl. the params_backbone group) during the Stage-1 '
                             'prototype-pretraining phase instead of keeping it frozen. Recommended when the '
                             'dataset is very different from the backbone\'s pretraining data (e.g. an ArcFace '
                             'face-recognition backbone on GMDB), where the default frozen pretrain stalls the '
                             'align loss. Pair with a larger --epochs_pretrain (e.g. 25).')
    parser.add_argument('--gallery_validation',
                        action='store_true',
                        help='After training, run gallery-based (retrieval) validation: build a '
                             'gallery from GMDB-frequent validation + 90%% of GMDB-rare patients and '
                             'evaluate top-1/5/10 retrieval accuracy on the held-out 10%% of rare '
                             'patients + frequent validation. See gallery_validation.py.')
    parser.add_argument('--gallery_test_frac',
                        type=float,
                        default=0.1,
                        help='Fraction of GMDB-rare patients held out as the test set in '
                             '--gallery_validation (the rest go into the gallery).')
    parser.add_argument('--post_validation',
                        type=str,
                        default='',
                        help='After training, run the standalone validation script(s) as '
                             'subprocesses against the finished run (--model_dir=<log_dir>). '
                             'Comma-separated subset of {gallery,binary,syndrome}, or "all". '
                             'Empty = none. E.g. --post_validation all')
    parser.add_argument('--reduced_syndromes',
                        action='store_true',
                        help='In --gallery_validation, restrict accuracy to the most-populated '
                             'syndromes: the top-275frequent and top-306 rare syndromes by distinct '
                             'image count, instead of all of them.')
    parser.add_argument('--masked_pool',
                        action='store_true',
                        help='Replace PIP-Net\'s global spatial max-pooling with region-masked '
                             'max-pooling: each prototype is assigned a facial region and its '
                             'presence score is the max activation only inside that region\'s binary '
                             'mask (built per image from MediaPipe landmarks). This structurally '
                             'prevents a prototype from contributing from outside its landmark region. '
                             'It switches the GMDB pipeline to landmark-safe (geometry-preserving) '
                             'augmentation so the landmarks stay valid for the augmented views.')
    parser.add_argument('--mask_radius',
                        type=float,
                        default=0.1,
                        help='Radius (as a fraction of the normalized image) of the disk drawn around '
                             'each region landmark when building the binary region mask for '
                             '--masked_pool. Larger = more permissive regions (closer to global pooling).')
    parser.add_argument('--loc_loss',
                        action='store_true',
                        help='Add a prototype-localization loss: each prototype is anchored to one '
                             'facial region (round-robin, as in --masked_pool) and its spatial '
                             'activation is pushed to lie INSIDE that region\'s landmark mask. Unlike '
                             '--masked_pool (a hard architectural mask) this is a soft training '
                             'penalty; the two can be used together or independently. Needs landmarks.')
    parser.add_argument('--loc_weight',
                        type=float,
                        default=2.0,
                        help='Weight of the --loc_loss term in the total loss.')
    parser.add_argument('--loc_temp',
                        type=float,
                        default=0.1,
                        help='Softmax temperature for the spatial attention in --loc_loss. Smaller = '
                             'sharper (the prototype is expected to peak in one place in its region).')
    parser.add_argument('--landmarks_dir',
                        type=str,
                        default='/data2/masino_lab/abamini/facial_data/Final_data/gmdb_landmarks_all/',
                        help='Directory with per-image MediaPipe landmark files named '
                             '"<image_id>_aligned_landmarks.txt" (one "id,x,y" per line).')
    parser.add_argument('--pretrain_landmarks_dir',
                        type=str,
                        default=None,
                        help='Directory with per-image landmark files for the Stage-1 prototype-pretraining '
                             'dataset (--pretrain_data_dir), supplying landmarks for region-masked pooling '
                             'during pretraining. E.g. CASIA-WebFace landmarks at '
                             '/data2/masino_lab/abamini/facial_data/casia_webface_align_landmarks. '
                             'Only used with --masked_pool. If omitted, pretraining uses global pooling.')
    parser.add_argument('--landmark_img_size',
                        type=int,
                        default=112,
                        help='Pixel size of the (square) image the landmark coordinates were computed on. '
                             'Used to normalize landmark coordinates to [0,1]. GMDB aligned faces are 112.')

    args = parser.parse_args()
    if args.run_number is None:
        args.run_number = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = f"{args.log_dir}_{args.run_number}_{args.net}_{args.masked_pool}"
    if len(args.log_dir.split('/'))>2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)


    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    
def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net:
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu.
                param.requires_grad = False
                # params_backbone.append(param)

    elif 'iresnet' in args.net:
        # Mirror the resnet50 freeze schedule: train the last block of layer4
        # (index .2 is the final block in IR-100's 3-block layer4), freeze
        # layer3/layer4 initially and unfreeze later, train layer2 at lr_net,
        # hard-freeze stem+layer1 to fit on one GPU. IR-100's layer3 has 30
        # blocks, so move layer3 to `else` if you hit OOM.
        for name, param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else:
                param.requires_grad = False

    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
          
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

