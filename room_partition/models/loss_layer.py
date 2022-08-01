import room_partition.models.utils.box_utils as box_utils
from torch import nn
import numpy as np
import torch
import math
import cv2


class LossFun(nn.Module):
    def __init__(self, device):
        super(LossFun, self).__init__()
        self.coverage = CoverageLoss(device)
        self.inside = InsideLoss(device)
        self.mutex = MutexLoss(device)
        self.device = device
        self.initial_mutex = 0

    def get_initial_mutex(self, pred_rect, boundary):
        boundary = boundary[:, :4]
        inside_boxes = get_inside_box(boundary)
        self.initial_mutex = self.mutex(pred_rect, inside_boxes)

    def forward(self, pred_rect, boundary, living_mask, opt=None):
        boundary = boundary[:, :4]
        inside_coords = get_inside_coords(boundary, living_mask)
        inside_boxes = get_inside_box(boundary)

        loss = 0
        if not opt == None and opt.coverage > 0:
            loss = loss + opt.coverage * self.coverage(pred_rect, inside_coords.to(self.device))
        if not opt == None and opt.inside > 0:
            loss = loss + opt.inside * self.inside(pred_rect, inside_boxes.to(self.device))
        if not opt == None and opt.mutex > 0:
            current_mutex = self.mutex(pred_rect, inside_boxes)
            if current_mutex > self.initial_mutex * 1.5:
                loss = loss + opt.mutex * (current_mutex - self.initial_mutex.to(self.device))
            else:
                loss = loss + 0

        return loss

    def obtain_inside_coords(self, boundary_id):
        b_id = boundary_id[:, -1]
        num = torch.unique(b_id).size(0)
        inside_coords = []
        for i_id in range(num):
            id_index = b_id == i_id
            boundary = boundary_id[id_index, :4]
            inside_coords.append(get_inside_coords(boundary.numpy()))
        return inside_coords

    def obtain_inside_box(self, boundary_id):
        b_id = boundary_id[:, -1]
        num = torch.unique(b_id).size(0)
        inside_boxes = []
        for i_id in range(num):
            id_index = b_id == i_id
            boundary = boundary_id[id_index, :4]
            inside_boxes.append(get_inside_box(boundary.numpy()))
        return inside_boxes


def get_inside_box(boundary, tensor=True):
    boundary = boundary[:, :2]
    boundary = boundary.astype(np.int)
    X, Y = np.linspace(0, 1, 128), np.linspace(0, 1, 128)
    x0, x1 = np.min(boundary[:, 0]), np.max(boundary[:, 0])
    y0, y1 = np.min(boundary[:, 1]), np.max(boundary[:, 1])
    box = np.array([[X[x0], Y[y0], X[x1], Y[y1]]])
    if tensor: box = torch.tensor(box).float()
    return box


def get_inside_coords(boundary, living_mask, size=(128, 128), tensor=True):
    h, w = size
    X = np.linspace(0, 1, w)
    Y = np.linspace(0, 1, h)
    img = np.zeros(size)
    boundary = boundary[:, :2]
    boundary = boundary * np.array(size) // 128
    boundary = boundary.astype(np.int)
    cv2.fillPoly(img, boundary.reshape(1, -1, 2), 1)
    img = img * (1 - living_mask.numpy())
    coords = np.where(img > 0)
    coords = np.stack((X[coords[1]], Y[coords[0]]), 1)
    if tensor: coords = torch.tensor(coords).unsqueeze(0).float()
    return coords


# fragment
# boundary

def cropped_box_iou(bboxes_pred, bboxes_gt, mask):
    H, W = mask.shape

    bboxes_pred = box_utils.norms_to_indices(bboxes_pred, W, H)
    bboxes_gt = box_utils.norms_to_indices(bboxes_gt, W, H)

    ious = torch.zeros(len(bboxes_pred)).to(bboxes_pred).float()
    for i in range(len(bboxes_pred)):
        box_pred = bboxes_pred[i]
        box_gt = bboxes_gt[i]

        mask_pred = mask.new_zeros(H, W)
        mask_gt = mask.new_zeros(H, W)

        mask_pred[box_pred[1]:box_pred[3], box_pred[0]:box_pred[2]] = 1
        mask_gt[box_gt[1]:box_gt[3], box_gt[0]:box_gt[2]] = 1

        mask_pred = mask_pred * mask
        mask_gt = mask_gt * mask

        union = (mask_pred + mask_gt)
        intersection = (mask_pred * mask_gt)

        ious[i] = len(intersection.nonzero()) / len(union.nonzero())
    return ious


def compose_by_cateogry(boxes, objs, category):
    H, W = category.shape
    image = torch.zeros_like(category).to(category)
    boxes = box_utils.norms_to_indices(boxes, H, W)

    overlaps = category.new_zeros(H, W, len(boxes))
    for i, box in enumerate(boxes):
        # overlap = image[box[1]:box[3],box[0]:box[2]].nonzero()+box[[1,0]]
        overlaps[box[1]:box[3], box[0]:box[2], i] = 1
        image[box[1]:box[3], box[0]:box[2]] = objs[i]

    overlap_vectors = overlaps.view(-1, len(boxes)).unique(dim=0)
    for vector in overlap_vectors:
        if vector.sum() < 2: continue
        overlap_region = (overlaps == vector).prod(-1).nonzero()
        unique, count = category[overlap_region[:, 0], overlap_region[:, 1]].unique(return_counts=True)
        overlap_objs = objs[vector.bool()]
        valid_u = [True if u in overlap_objs else False for u in unique]
        count = count[valid_u]
        unique = unique[valid_u]
        if len(unique) > 0:
            winner = unique[count.argmax()]
            image[overlap_region[:, 0], overlap_region[:, 1]] = winner
    return image


def sample_fragment(step=2):
    """
    Parameters:
    ----------
    step: int, sample step in linspace [0,1]

    Returns:
    ----------
    ret: [step^step,2]
    """
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, step),
            torch.linspace(0, 1, step)
        )
        , dim=-1).reshape(-1, 2)


def sample_boundary(step=2):
    """
    Parameters:
    ----------
    step: int, sample step in linspace [0,1]

    Returns:
    ----------
    ret: [step*4,2]
    """
    return torch.cat([
        torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 2),
            torch.linspace(0, 1, step)
        ), dim=-1).reshape(-1, 2),
        torch.stack(torch.meshgrid(
            torch.linspace(0, 1, step),
            torch.linspace(0, 1, 2)
        ), dim=-1).reshape(-1, 2)
    ])


def fragment_outside_box(fragments, boxes):
    """
    if points in fragment outside box

    Calculate line distance among points of fragment i and lines of box j, get [P,B,4]
    if all 4 values of a point are great than or equal to 0, the point is inside the box
    else the point is in outside the box

    Parameters:
    ----------
    fragments: [F,FP,2]
    boxes: [B,4]

    Return:
    ----------
    ret: [F,FP,B]
    """
    assert fragments.dim() == 3
    F, FP, _ = fragments.shape
    B, _ = boxes.shape

    diff = torch.cat([
        fragments.view(F, FP, 1, 2) - boxes[:, :2].view(1, 1, B, 2),
        boxes[:, 2:].view(1, 1, B, 2) - fragments.view(F, FP, 1, 2)
    ], dim=-1)

    return ((diff >= 0).sum(-1) != 4).float()


def fragment_box_distance(fragments, box_points):
    """
    calcuate distance among fragments of fragmaent_i and box_points of box_j
    get the smallest distance for each point of box_i to box_j

    Parameters:
    ----------
    fragments: [F,FP,2]
    box_points: [B,BP,2]

    Return:
    ----------
    ret: [F,FP,B]
    """
    F, FP, _ = fragments.shape
    B, BP, _ = box_points.shape
    # [F,FP,B,BP] -> [F,FP,B]
    # return (fragments.view(F,FP,1,1,2)-box_points.view(1,1,B,BP,2)).norm(dim=-1).min(-1)[0]
    return (fragments.view(F, FP, 1, 1, 2) - box_points.view(1, 1, B, BP, 2)).pow(2).sum(-1).min(-1)[0]


def coverage_loss(boxes, fragments, step=2):
    """
    boxes can cover all points in fragments

    Parameters:
    ----------
    boxes: [B,4], NBoxes with (x0,y0,x1,y1)
    fragments: [F,FP,2], NBox wit (x,y)
    """
    BP = step * 4
    B, _ = boxes.shape
    F, FP, _ = fragments.shape
    box_wh = boxes[:, 2:] - boxes[:, :2]

    # [B,BP,2]
    box_points = sample_boundary(step=step).view(1, BP, 2) * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
    # [F,FP,B]
    f_out_box = fragment_outside_box(fragments, boxes)
    # [F,FP,B]
    f_b_dist = fragment_box_distance(fragments, box_points)
    # [F,FP]
    return (f_b_dist * f_out_box).min(-1)[0].sum() / FP


def inside_loss(boxes, fragment_boxes, step=2):
    """
    all points on the boundary of boxes are inside fragment_boxes

    Parameters:
    ----------
    boxes: [B,4], NBoxes with (x0,y0,x1,y1)
    fragment_boxes: [F,4], NBox wit (x,y)
    """
    B, _ = boxes.shape
    P = step * 4
    F, _ = fragment_boxes.shape
    box_wh = boxes[:, 2:] - boxes[:, :2]
    fragment_wh = fragment_boxes[:, 2:] - fragment_boxes[:, :2]

    # [B,BP,2]
    box_fragments = sample_boundary(step=step).view(1, P, 2) * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
    # [F,FP,2]
    fragment_boundaries = sample_boundary(step=step).view(1, P, 2) * fragment_wh.view(F, 1, 2) + fragment_boxes[:,
                                                                                                 :2].view(F, 1, 2)
    # [B,BP,F]
    f_out_box = fragment_outside_box(box_fragments, fragment_boxes)
    # [B,BP,F]
    f_b_dist = fragment_box_distance(box_fragments, fragment_boundaries)
    # [B,BP]
    return (f_b_dist * f_out_box).sum() / (B * P)


def mutex_loss(boxes, step=2):
    """
    sum of min-pixel-boundary distance / sum of pixels in boxes

    Parameters:
    ----------
    boxes: B*4, bboxes with (x0,y0,x1,y1)
    ref_points: P*2, bbox wit (x,y)
    """
    B = boxes.shape[0]
    BP = step * 4
    FP = step * step
    box_wh = boxes[:, 2:] - boxes[:, :2]

    # [B,FP,2]
    fragments = sample_fragment(step=step).view(1, FP, 2) * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
    # [B,BP,2]
    box_points = sample_boundary(step=step).view(1, BP, 2) * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)

    # [B,FP,B]
    f_out_box = fragment_outside_box(fragments, boxes)
    # [B,FP,B] B个Box的PP个点与B个Box的最小距离
    f_b_dist = fragment_box_distance(fragments, box_points)

    return (f_b_dist * f_out_box).sum() / (B * FP - B)


class InsideLoss(nn.Module):
    def __init__(self, device, nsample=400):
        super(InsideLoss, self).__init__()
        self.device = device
        self.bstep = round(nsample / 4)
        self.fstep = round(math.sqrt(nsample))
        self.BP = self.bstep * 4
        self.FP = self.fstep * self.fstep
        self.boundary = sample_boundary(step=self.bstep).view(1, self.BP, 2)
        self.fragment = sample_fragment(step=self.fstep).view(1, self.FP, 2)

        self.boundary = self.boundary.to(device)
        self.fragment = self.fragment.to(device)

    def _inside_loss(self, boxes, fragment_boxes):
        B, _ = boxes.shape
        F, _ = fragment_boxes.shape
        box_wh = boxes[:, 2:] - boxes[:, :2]
        fragment_wh = fragment_boxes[:, 2:] - fragment_boxes[:, :2]

        # [B,FP,2]
        box_fragments = self.fragment * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
        # [F,BP,2]
        fragment_boundaries = self.boundary * fragment_wh.view(F, 1, 2) + fragment_boxes[:, :2].view(F, 1, 2)
        # [B,FP,F]
        f_out_box = fragment_outside_box(box_fragments, fragment_boxes)
        # [B,FP,F]
        f_b_dist = fragment_box_distance(box_fragments, fragment_boundaries)
        # [B,FP]
        return (f_b_dist * f_out_box).sum() / (B * self.FP)

    def test(self, boxes, fragment_boxes):
        with torch.no_grad():
            return self._inside_loss(boxes, fragment_boxes)

    def forward(self, pred, fragment_boxes, reduction="mean"):
        losses = []
        loss = self._inside_loss(pred, fragment_boxes)
        losses.append(loss)
        return torch.mean(torch.stack(losses))


class CoverageLoss(nn.Module):
    def __init__(self, device, nsample=400):
        super(CoverageLoss, self).__init__()
        self.step = round(nsample / 4)
        self.BP = self.step * 4
        self.device = device
        self.boundary = sample_boundary(step=self.step).view(1, self.BP, 2)
        self.boundary = self.boundary.to(device)

    def _coverage_loss(self, boxes, fragments):
        B, _ = boxes.shape
        F, FP, _ = fragments.shape
        box_wh = boxes[:, 2:] - boxes[:, :2]

        # [B,BP,2]
        box_points = self.boundary * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
        # [F,FP,B]
        f_out_box = fragment_outside_box(fragments, boxes)
        # [F,FP,B]
        f_b_dist = fragment_box_distance(fragments, box_points)
        # [F,FP]
        return (f_b_dist * f_out_box).min(-1)[0].sum() / FP

    def test(self, boxes, fragments):
        with torch.no_grad():
            return self._coverage_loss(boxes, fragments)

    def forward(self, pred, fragments, reduction="mean"):
        losses = []
        loss = self._coverage_loss(pred, fragments)
        losses.append(loss)
        return torch.mean(torch.stack(losses))


class MutexLoss(nn.Module):
    def __init__(self, device, nsample=400):
        super(MutexLoss, self).__init__()
        self.bstep = round(nsample / 4)
        self.fstep = round(math.sqrt(nsample))
        self.BP = self.bstep * 4
        self.FP = self.fstep * self.fstep
        self.device = device
        self.boundary = sample_boundary(step=self.bstep).view(1, self.BP, 2)
        self.fragment = sample_fragment(step=self.fstep).view(1, self.FP, 2)

        self.boundary = self.boundary.to(device)
        self.fragment = self.fragment.to(device)

    def _mutex_loss(self, boxes):
        B = boxes.shape[0]
        box_wh = boxes[:, 2:] - boxes[:, :2]

        # [B,FP,2]
        fragments = self.fragment * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)
        # [B,BP,2]
        box_points = self.boundary * box_wh.view(B, 1, 2) + boxes[:, :2].view(B, 1, 2)

        # [B,FP,B]
        f_in_box = 1 - fragment_outside_box(fragments, boxes)
        # clear dist to self
        f_in_box[range(B), :, range(B)] = 0
        # [B,FP,B] B个Box的PP个点与B个Box的最小距离
        f_b_dist = fragment_box_distance(fragments, box_points)

        return (f_b_dist * f_in_box).sum() / (B * self.FP - B)

    def test(self, boxes):
        with torch.no_grad():
            return self._mutex_loss(boxes)

    def forward(self, pred, objs=None, reduction="mean"):
        losses = []
        loss = self._mutex_loss(pred)
        losses.append(loss)
        return torch.mean(torch.stack(losses))


if __name__ == "__main__":
    # [4]
    fragment_box = torch.tensor(
        [0.25, 0.25, 0.75, 0.75]
    ).view(1, 4)
    # [B,4]
    boxes = torch.tensor([
        [0.25, 0.00, 0.50, 0.50],
        [0.50, 0.50, 1.00, 0.75]
    ], requires_grad=True)
    # (0.25**2*2+(0.25*sqrt(2))**2*2)/(8*2)=0.046875
    loss = inside_loss(boxes, fragment_box, step=2)
    print(loss)
    insideL = InsideLoss(cuda=False)
    print(insideL(boxes, fragment_box, torch.tensor([0, 0])))

    boxes = torch.tensor([
        [0.25, 0.00, 0.50, 0.50],
        [0.50, 0.50, 1.00, 0.75]
    ], requires_grad=True)
    #
    X = torch.linspace(0., 1., 4)
    Y = torch.linspace(0., 1., 4)
    # P,2
    fragment_points = torch.tensor([(X[x], Y[y]) for x in range(1, 3) for y in range(1, 3)]).view(1, 4, 2)
    # (0.09**2+0.16**2+0.16**2*2)/4=0.021225
    loss = coverage_loss(boxes, fragment_points)
    print(loss)
    coverageL = CoverageLoss(cuda=False)
    print(coverageL(boxes, fragment_points, torch.tensor([0, 0])))

    # B,4
    boxes = torch.tensor([
        [0.25, 0.00, 0.49, 0.49],  # 0
        [0.00, 0.25, 0.49, 0.49],  # 1
        [0.51, 0.51, 1.00, 0.75],  # 2
        [0.51, 0.51, 0.75, 1.00]  # 3
    ], requires_grad=True)

    # (( ( (0.24**2+0.25**2) + (0.51**2+0.26**2)*2 )
    # +(0.25**2+ (0.51**2+0.02**2)*2 )
    # +((0.26**2+0.02**2)*2)
    # +((0.02**2*2)*2))*4)
    # /(4*4-4) = 0.4988666
    loss = mutex_loss(boxes, step=2)
    print(loss)
    mutexL = MutexLoss(cuda=False)
    print(mutexL(boxes, torch.tensor([0, 0, 1, 1])))
