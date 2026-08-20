from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math

# Facial regions defined by representative MediaPipe Face Mesh landmark indices
# (468 mesh + iris 468-477). Used by --masked_pool (see MaskedMaxPool in pipnet.py):
# each prototype is assigned to one region and its presence score is the max activation
# only inside that region's landmark-derived mask.
REGION_LANDMARKS = { 'left_eye': [468, 469, 470, 471, 472, 33, 133, 159, 145], 
                    'right_eye': [473, 474, 475, 476, 477, 263, 362, 386, 374], 
                    'left_brow': [105, 66, 107, 55, 65], 
                    'right_brow': [334, 296, 336, 285, 295], 
                    'nose': [1, 2, 4, 5, 6, 19, 94], 
                    'left_cheek': [50, 116, 117, 118, 205], 
                    'right_cheek': [280, 345, 346, 347, 425], 
                    'mouth': [13, 14, 61, 291, 0, 17, 78, 308], 
                    'chin': [152, 175, 199, 148, 377], 
                    'forehead': [10, 9, 151, 8, 107, 336], 
                    }


def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, batch in train_iter:
        if len(batch) == 4:
            xs1, xs2, ys, lms = batch
            lms = lms.to(device)
        else:
            xs1, xs2, ys = batch
            lms = None

        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a forward pass through the network. The landmarks are duplicated to
        # match the two concatenated augmented views (geometry-preserving augmentation
        # keeps them valid for both); the model uses them only for region-masked pooling.
        lms_fwd = torch.cat([lms, lms], dim=0) if lms is not None else None
        proto_features, pooled, out = net(torch.cat([xs1, xs2]), lms=lms_fwd)

        # Optional prototype-localization loss (--loc_loss): needs landmarks. Config is
        # attached to net.module in main.py when the flag is set.
        loc_val = None
        if getattr(net.module, '_loc_loss', False) and lms_fwd is not None:
            loc_val = localization_loss(proto_features, lms_fwd, net.module._loc_region_idx,
                                        net.module._loc_radius, net.module._loc_temp)

        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, net.module._classification.normalization_multiplier, pretrain, finetune, criterion, train_iter, loc_val=loc_val, loc_weight=getattr(net.module, '_loc_weight', 0.), print=True, EPS=1e-8)
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0))
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info

def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, loc_val=None, loc_weight=0., print=True, EPS=1e-10):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        loss += t_weight * tanh_loss
        if loc_val is not None:                 # prototype-localization loss (--loc_loss)
            loss += loc_weight * loc_val

    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)

        if finetune:
            loss= cl_weight * class_loss
        else:
            loss+= cl_weight * class_loss

    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc=0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    if print:
        with torch.no_grad():
            loc_str = f', LLoc:{loc_val.item():.4f}' if loc_val is not None else ''
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}{loc_str}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                train_iter.set_postfix_str(
                f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}{loc_str}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
    return loss, acc

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# --- prototype-localization loss (--loc_loss) ------------------------------
def build_region_masks(lms, H, W, region_landmark_indices, radius):
    """Boolean (B, R, H, W) region-membership mask at the feature-map resolution: a grid cell
    belongs to region g if it lies within `radius` (fraction of the normalized image) of any of
    that region's landmarks. Mirrors MaskedMaxPool._build_region_masks."""
    device, dtype = lms.device, lms.dtype
    B = lms.shape[0]
    rows = torch.linspace(0., 1., steps=H, device=device, dtype=dtype)
    cols = torch.linspace(0., 1., steps=W, device=device, dtype=dtype)
    gr, gc = torch.meshgrid(rows, cols, indexing='ij')
    grid = torch.stack([gr.reshape(-1), gc.reshape(-1)], dim=-1)          # (H*W, 2)
    r2 = radius ** 2
    masks = []
    for idx in region_landmark_indices:
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        pts = lms[:, idx_t, :]                                            # (B, k, 2)
        valid = (pts >= 0).all(dim=-1)                                    # (B, k)
        d2 = ((grid[None, None] - pts[:, :, None]) ** 2).sum(dim=-1)      # (B, k, H*W)
        within = (d2 <= r2) & valid[:, :, None]
        masks.append(within.any(dim=1).reshape(B, H, W))                 # (B, H, W)
    return torch.stack(masks, dim=1)                                      # (B, R, H, W)


def localization_loss(proto_features, lms, region_landmark_indices, radius, temp, EPS=1e-8):
    """Anchor each prototype's spatial activation inside its assigned facial region.

    Prototype d is mapped round-robin to region r(d) = d mod R (matching MaskedMaxPool). We take
    the spatial softmax (temperature `temp`) of prototype d's H*W activation map, sum the
    probability mass falling INSIDE region r(d)'s landmark mask, and penalize -log(mass) so the
    prototype is pushed to fire within its region. Prototypes whose region mask is empty for an
    image (missing landmarks) are skipped, so a missing landmark simply contributes no loss."""
    B, P, H, W = proto_features.shape
    R = len(region_landmark_indices)
    region_masks = build_region_masks(lms, H, W, region_landmark_indices, radius)   # (B,R,H,W)
    proto2region = torch.arange(P, device=proto_features.device) % R
    proto_masks = region_masks[:, proto2region].flatten(2).float()                  # (B,P,H*W)
    attn = torch.softmax(proto_features.flatten(2) / temp, dim=2)                    # (B,P,H*W)
    inside = (attn * proto_masks).sum(dim=2)                                         # (B,P)
    valid = proto_masks.any(dim=2)                                                   # (B,P)
    if valid.sum() == 0:
        return proto_features.new_zeros(())
    return (-torch.log(inside + EPS))[valid].mean()


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss