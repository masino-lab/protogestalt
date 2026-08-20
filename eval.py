import argparse
import os


#python eval.py --checkpoint /data2/masino_lab/abamini/pipnet/runs/run_pipnet_20260515_115308/checkpoints/net_trained_last --log_dir runs/eval_output --dataset gmdb --net convnext_tiny_26 --image_size 112


import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

from pipnet.pipnet import PIPNet, get_network
from pipnet.test import eval_pipnet
from util.data import get_dataloaders
from util.log import Log


def get_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Evaluate a trained PIP-Net on the test set')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint, e.g. runs/run_pipnet_xxx/checkpoints/net_trained_last')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory to write evaluation results')

    # Must match the values used during training
    parser.add_argument('--dataset', type=str, default='gmdb')
    parser.add_argument('--net', type=str, default='convnext_tiny_26')
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--disable_pretrained', action='store_true')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_pretrain', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--validation_size', type=float, default=0.)
    parser.add_argument('--weighted_loss', action='store_true')

    # Hardware
    parser.add_argument('--gpu_ids', type=str, default='')
    parser.add_argument('--disable_cuda', action='store_true')

    # Required by get_dataloaders but unused during eval
    parser.add_argument('--dir_for_saving_images', type=str, default='visualization_results')

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    args.wshape = 0  # filled after first forward pass

    return args


def main():
    args = get_eval_args()

    # Device setup (mirrors main.py)
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids != '':
        for m in gpu_list:
            device_ids.append(int(m))

    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device('cuda')
            device_ids.append(torch.cuda.current_device())
        else:
            device = torch.device('cuda:' + str(device_ids[0]))
    else:
        device = torch.device('cpu')

    print('Device:', device, flush=True)

    # Data — only testloader and projectloader are used
    _, _, _, _, projectloader, testloader, _, classes = get_dataloaders(args, device)
    print('Classes:', testloader.dataset.class_to_idx, flush=True)

    # Network
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
    net = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('Checkpoint loaded from', args.checkpoint, flush=True)

    # Resolve wshape
    with torch.no_grad():
        net.eval()
        xs, _ = next(iter(testloader))
        proto_features, _, _ = net(xs.to(device))
        args.wshape = proto_features.shape[-1]

    # Log object required by eval_pipnet signature
    log = Log(args.log_dir)

    # Run evaluation
    net.eval()
    eval_info = eval_pipnet(net, testloader, 'eval', device, log)

    y_trues = eval_info['y_trues']
    y_preds = eval_info['y_preds_classes']
    y_scores = eval_info['y_preds_scores']
    class_names = list(testloader.dataset.class_to_idx.keys())
    is_binary = net.module._num_classes == 2

    # Compile metrics
    lines = []
    lines.append('=' * 60)
    lines.append('PIPNet Evaluation Results')
    lines.append(f'Checkpoint : {args.checkpoint}')
    lines.append(f'Dataset    : {args.dataset}')
    lines.append('=' * 60)

    lines.append(f'\nTop-1 accuracy : {eval_info["top1_accuracy"]:.4f}')
    if is_binary:
        lines.append(f'F1 score       : {eval_info["top5_accuracy"]:.4f}')
    else:
        lines.append(f'Top-5 accuracy : {eval_info["top5_accuracy"]:.4f}')
        lines.append(f'Top-10 accuracy: {eval_info["top10_accuracy"]:.4f}')

    lines.append(f'Balanced accuracy: {balanced_accuracy_score(y_trues, y_preds):.4f}')

    lines.append('\n--- Per-class report ---')
    lines.append(classification_report(y_trues, y_preds, target_names=class_names, digits=4))

    lines.append('--- Confusion matrix ---')
    lines.append('Rows = true class, Columns = predicted class')
    lines.append('Classes: ' + ', '.join(f'{i}={n}' for i, n in enumerate(class_names)))
    lines.append(str(confusion_matrix(y_trues, y_preds)))

    try:
        if is_binary:
            auc = roc_auc_score(y_trues, y_scores)
            lines.append(f'\nAUC: {auc:.4f}')
        else:
            auc_macro = roc_auc_score(y_trues, y_scores, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_trues, y_scores, multi_class='ovr', average='weighted')
            lines.append(f'\nAUC (macro)   : {auc_macro:.4f}')
            lines.append(f'AUC (weighted): {auc_weighted:.4f}')
    except ValueError as e:
        lines.append(f'\nAUC could not be computed: {e}')

    lines.append(f'\nNum non-zero prototypes: {eval_info["num non-zero prototypes"]}')
    lines.append(f'Almost-nonzeros (pooled): {eval_info["almost_nonzeros"]:.4f}')
    lines.append(f'Local size (all classes): {eval_info["local_size_all_classes"]:.4f}')

    report = '\n'.join(lines)
    print(report, flush=True)

    out_path = os.path.join(args.log_dir, 'eval_metrics.txt')
    with open(out_path, 'w') as f:
        f.write(report + '\n')

    print(f'\nMetrics saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
