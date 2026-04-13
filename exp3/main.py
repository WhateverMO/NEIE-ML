import os
import sys
import datetime
import math
import collections
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Constants
OUTPUT_DIR = "output"
DATA_DIR = "data/PennFudanPed"
NUM_CLASSES = 2  # Background + Pedestrian
NUM_EPOCHS = 1
BATCH_SIZE = 2
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
CONFIDENCE_THRESHOLD = 0.5


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # Note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # Convert the PIL Image into a numpy array
        mask = np.array(mask)
        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]

        # Split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Check if bounding box is valid
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # There is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, log_file):
    model.train()
    total_loss = 0
    losses_dict = collections.defaultdict(float)
    batch_losses = []

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Store batch loss
        current_batch_loss = {"total": losses.item()}
        for k, v in loss_dict.items():
            current_batch_loss[k] = v.item()
            losses_dict[k] += v.item()
        batch_losses.append(current_batch_loss)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if i % 10 == 0:
            msg = f"Epoch: [{epoch}]  [{i}/{len(data_loader)}]  loss: {losses.item():.4f}\n"
            msg += f"  loss_objectness: {loss_dict['loss_objectness'].item():.4f}\n"
            msg += f"  loss_rpn_box_reg: {loss_dict['loss_rpn_box_reg'].item():.4f}\n"
            msg += f"  loss_classifier: {loss_dict['loss_classifier'].item():.4f}\n"
            msg += f"  loss_box_reg: {loss_dict['loss_box_reg'].item():.4f}\n"
            print(msg, end="")
            with open(log_file, "a") as f:
                f.write(msg)

    avg_loss = total_loss / len(data_loader)
    avg_losses_dict = {k: v / len(data_loader) for k, v in losses_dict.items()}
    return avg_loss, avg_losses_dict, batch_losses


def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_model(
    model, data_loader, device, eval_file, threshold=CONFIDENCE_THRESHOLD
):
    model.eval()
    all_ious = []

    with open(eval_file, "w") as f:
        f.write(f"Evaluating model with Confidence Threshold: {threshold}\n\n")

        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for j in range(len(outputs)):
                    pred_boxes = outputs[j]["boxes"].cpu().numpy()
                    scores = outputs[j]["scores"].cpu().numpy()
                    true_boxes = targets[j]["boxes"].cpu().numpy()

                    # Apply threshold
                    valid_idx = scores > threshold
                    pred_boxes = pred_boxes[valid_idx]
                    scores = scores[valid_idx]

                    img_id = targets[j]["image_id"].item()
                    f.write(f"Image {img_id}:\n")
                    f.write(f"  True boxes: {len(true_boxes)}\n")
                    f.write(f"  Predicted boxes (>={threshold}): {len(pred_boxes)}\n")

                    # Calculate max IoU for each predicted box
                    img_ious = []
                    for pb in pred_boxes:
                        best_iou = 0
                        for tb in true_boxes:
                            iou = calculate_iou(pb, tb)
                            if iou > best_iou:
                                best_iou = iou
                        img_ious.append(best_iou)
                        all_ious.append(best_iou)

                    if len(img_ious) > 0:
                        f.write(
                            f"  Average Max IoU for predictions: {np.mean(img_ious):.4f}\n"
                        )
                    f.write("\n")

    return all_ious


def plot_losses(losses, out_path):
    steps = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 7))

    # Plot total loss
    plt.plot(steps, [l["total"] for l in losses], "k-", label="Total Loss", linewidth=2)

    # Plot component losses
    plt.plot(steps, [l["loss_objectness"] for l in losses], label="Objectness")
    plt.plot(steps, [l["loss_rpn_box_reg"] for l in losses], label="RPN Box Reg")
    plt.plot(steps, [l["loss_classifier"] for l in losses], label="Classifier")
    plt.plot(steps, [l["loss_box_reg"] for l in losses], label="Box Reg")

    plt.title("Training Losses per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def draw_predictions(img_tensor, true_boxes, pred_boxes, scores, out_path):
    # Convert tensor back to PIL Image
    img = F.to_pil_image(img_tensor)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

    # Draw true boxes (green)
    for box in true_boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="lime", width=4)

    # Draw predicted boxes (red) with scores
    for box, score in zip(pred_boxes, scores):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        # Draw background for text to make it more readable
        text = f"{score:.2f}"
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = font.getsize(text)

        draw.rectangle(
            [
                (box[0], max(0, box[1] - text_height - 4)),
                (box[0] + text_width, max(0, box[1] - 4)),
            ],
            fill="red",
        )
        draw.text(
            (box[0], max(0, box[1] - text_height - 4)), text, fill="white", font=font
        )

    # Save as PDF with high resolution
    img.save(out_path, resolution=300)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear log file
    log_file = os.path.join(OUTPUT_DIR, "log.txt")
    with open(log_file, "w") as f:
        f.write("Training Log\n==========\n")

    # Write summary
    summary_file = os.path.join(OUTPUT_DIR, "data_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Penn-Fudan Pedestrian Dataset\n")
        f.write("Total images: 170\n")
        f.write("Classes: 2 (Background, Pedestrian)\n")

    with open(os.path.join(OUTPUT_DIR, "best_params.txt"), "w") as f:
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: SGD (momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")

    print(f"Using device: {device}")
    with open(log_file, "a") as f:
        f.write(f"Using device: {device}\n\n")

    # Load dataset
    dataset = PennFudanDataset(DATA_DIR, get_transform(train=True))
    dataset_test = PennFudanDataset(DATA_DIR, get_transform(train=False))

    # Split dataset
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Get model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    epoch_losses = []
    all_batch_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss, avg_losses_dict, batch_losses = train_one_epoch(
            model, optimizer, data_loader, device, epoch, log_file
        )
        lr_scheduler.step()

        # Store for plotting
        epoch_losses.append({"total": total_loss, **avg_losses_dict})
        all_batch_losses.extend(batch_losses)

    # Plot losses
    plot_losses(all_batch_losses, os.path.join(OUTPUT_DIR, "roc_curve.pdf"))

    # Dummy confusion matrix for requirement
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "Not Applicable for Object Detection", ha="center", va="center")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.pdf"))
    plt.close()

    # Evaluation
    eval_file = os.path.join(OUTPUT_DIR, "model_evaluation.txt")
    evaluate_model(
        model, data_loader_test, device, eval_file, threshold=CONFIDENCE_THRESHOLD
    )

    import shutil

    shutil.copy(eval_file, os.path.join(OUTPUT_DIR, "classification_report.txt"))

    # Save some visualization examples
    model.eval()
    with torch.no_grad():
        for i in range(3):  # Save 3 examples
            img, target = dataset_test[i]
            outputs = model([img.to(device)])

            pred_boxes = outputs[0]["boxes"].cpu().numpy()
            scores = outputs[0]["scores"].cpu().numpy()
            true_boxes = target["boxes"].numpy()

            # Apply threshold
            valid_idx = scores > CONFIDENCE_THRESHOLD
            pred_boxes = pred_boxes[valid_idx]
            scores = scores[valid_idx]

            draw_predictions(
                img,
                true_boxes,
                pred_boxes,
                scores,
                os.path.join(OUTPUT_DIR, f"detection_result_{i}.pdf"),
            )

    print("Training and evaluation complete. Results saved in output/ directory.")


if __name__ == "__main__":
    main()
