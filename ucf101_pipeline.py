"""
UC F101 processing + baseline CNN training pipeline

Usage (examples):
  python ucf101_pipeline.py --check
  python ucf101_pipeline.py --extract-frames --frame-rate 10 --outdir ./frames
  python ucf101_pipeline.py --prepare-dataloaders --frames-dir ./frames --frames-per-clip 16
  python ucf101_pipeline.py --train-cnn --epochs 10 --batch-size 8 --save-path cnn_baseline.pth
  python ucf101_pipeline.py --evaluate --checkpoint cnn_baseline.pth

Notes:
- This script provides modular functions for dataset validation, optional cleaning, frame extraction,
  DataLoader preparation (using torchvision.datasets.UCF101 or from frames), a simple 3D CNN baseline,
  training loop with checkpointing and evaluation utilities.
- You must install: torch torchvision opencv-python tqdm scikit-learn tensorboard

"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------- Configuration / Helpers --------------------

SUPPORTED_VIDEO_EXT = {'.avi', '.mp4', '.mov', '.mkv'}


def is_video_file(p: Path):
    return p.suffix.lower() in SUPPORTED_VIDEO_EXT


def validate_ucf_structure(root: str, ann_path: str):
    """Check that video root exists, annotation files exist and report basic counts."""
    r = Path(root)
    a = Path(ann_path)
    if not r.exists():
        raise FileNotFoundError(f"UCF root not found: {root}")
    if not a.exists():
        raise FileNotFoundError(f"Annotation path not found: {ann_path}")

    # count classes (folders) and total videos
    class_dirs = [d for d in r.iterdir() if d.is_dir()]
    total_videos = 0
    for d in class_dirs:
        vids = [f for f in d.iterdir() if is_video_file(f)]
        total_videos += len(vids)
    print(f"Found {len(class_dirs)} class folders and {total_videos} video files under {root}")
    print(f"Annotation folder contents: {list(a.iterdir())[:20]}")
    return len(class_dirs), total_videos


# -------------------- Cleaning utilities --------------------

def check_and_move_corrupt(root: str, corrupt_out: str = None, try_open=True):
    """Scan videos and move unreadable/corrupt ones to corrupt_out.
       Returns list of moved files."""
    moved = []
    root_p = Path(root)
    corrupt_out_p = Path(corrupt_out) if corrupt_out else root_p / "_corrupt_videos"
    corrupt_out_p.mkdir(parents=True, exist_ok=True)

    for cls in tqdm(sorted([d for d in root_p.iterdir() if d.is_dir()]), desc="Checking classes"):
        for vid in cls.iterdir():
            if not is_video_file(vid):
                continue
            try:
                if try_open:
                    cap = cv2.VideoCapture(str(vid))
                    ok, _ = cap.read()
                    cap.release()
                    if not ok:
                        raise ValueError("cv2 failed to read first frame")
                # if we reach here, assume ok
            except Exception as e:
                target = corrupt_out_p / cls.name
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(vid), str(target / vid.name))
                moved.append(str(vid))
    print(f"Moved {len(moved)} corrupt/unreadable videos to {corrupt_out_p}")
    return moved


# -------------------- Frame extraction --------------------

def extract_frames_from_video(video_path: str, out_dir: str, frame_rate: int = 10, resize=(112,112)):
    """Save every Nth frame (frame_rate) from video to output dir."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1
    cap.release()
    return saved


def extract_frames_dataset(root: str, out_root: str, frame_rate: int = 10, resize=(112,112), classes=None, max_videos_per_class=None):
    """Walk through class folders in root and extract frames into out_root/<class>/video_x/frames..."""
    root_p = Path(root)
    out_p = Path(out_root)
    out_p.mkdir(parents=True, exist_ok=True)
    classes_to_process = sorted([d for d in root_p.iterdir() if d.is_dir()])
    if classes:
        classes_to_process = [d for d in classes_to_process if d.name in classes]

    for cls in tqdm(classes_to_process, desc="Classes"):
        vids = [v for v in cls.iterdir() if is_video_file(v)]
        if max_videos_per_class:
            vids = vids[:max_videos_per_class]
        for vid in tqdm(vids, desc=f"Processing {cls.name}", leave=False):
            video_id = vid.stem
            out_dir = out_p / cls.name / video_id
            if out_dir.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = extract_frames_from_video(str(vid), str(out_dir), frame_rate=frame_rate, resize=resize)
    print("Frame extraction completed")


# -------------------- DataLoaders --------------------

def make_ucf_dataloaders(root: str, ann_path: str, frames_per_clip: int = 16, batch_size=8, num_workers=4):
    """Use torchvision.datasets.UCF101 to create dataloaders (video clips returned as tensors).
       Returned tensors have shape [B, C, T, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor()
    ])
    train_ds = datasets.UCF101(root=root, annotation_path=ann_path, frames_per_clip=frames_per_clip,
                               step_between_clips=1, train=True, transform=transform)
    test_ds = datasets.UCF101(root=root, annotation_path=ann_path, frames_per_clip=frames_per_clip,
                              step_between_clips=1, train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    return train_loader, test_loader


# -------------------- Simple 3D CNN baseline --------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        # compute flattened size for fc, but we will do adaptive pooling to keep it simple
        self.adaptive = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.fc = nn.Linear(32 * 1 * 4 * 4, num_classes)

    def forward(self, x):
        # expects [B, C, T, H, W]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.adaptive(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -------------------- Training / Evaluation --------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    preds = []
    trues = []
    for videos, _, labels in tqdm(loader, desc="Train batches"):
        # videos shape from UCF101: [B, T, C, H, W] -> convert to [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    return np.mean(losses), acc


def evaluate_model(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for videos, _, labels in tqdm(loader, desc="Eval batches"):
            videos = videos.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    return np.mean(losses), acc, cm


# -------------------- CLI / main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='UCF101', help='UCF101 videos root')
    parser.add_argument('--ann', type=str, default='UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist', help='annotation folder')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--extract-frames', action='store_true')
    parser.add_argument('--frames-out', type=str, default='frames', help='frames output folder')
    parser.add_argument('--frame-rate', type=int, default=10, help='save every Nth frame')
    parser.add_argument('--frames-per-clip', type=int, default=16)
    parser.add_argument('--prepare-dataloaders', action='store_true')
    parser.add_argument('--train-cnn', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-path', type=str, default='cnn_baseline.pth')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.check:
        validate_ucf_structure(args.root, args.ann)

    if args.clean:
        check_and_move_corrupt(args.root)

    if args.extract_frames:
        extract_frames_dataset(args.root, args.frames_out, frame_rate=args.frame_rate, resize=(112,112))

    if args.prepare_dataloaders:
        train_loader, test_loader = make_ucf_dataloaders(args.root, args.ann, frames_per_clip=args.frames_per_clip,
                                                         batch_size=args.batch_size)
        # simple peek
        for videos, _, labels in train_loader:
            print("Loaded batch video tensor shape:", videos.shape)
            break

    if args.train_cnn:
        train_loader, test_loader = make_ucf_dataloaders(args.root, args.ann, frames_per_clip=args.frames_per_clip,
                                                         batch_size=args.batch_size)
        device = torch.device(args.device)
        model = Simple3DCNN(num_classes=101).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        for ep in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _ = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'acc': best_acc}, args.save_path)
                print(f"Saved best checkpoint to {args.save_path}")

    if args.evaluate:
        if not args.checkpoint:
            print("Provide --checkpoint to evaluate")
            return
        device = torch.device(args.device)
        train_loader, test_loader = make_ucf_dataloaders(args.root, args.ann, frames_per_clip=args.frames_per_clip,
                                                         batch_size=args.batch_size)
        chk = torch.load(args.checkpoint, map_location=device)
        model = Simple3DCNN(num_classes=101).to(device)
        model.load_state_dict(chk['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, cm = evaluate_model(model, test_loader, criterion, device)
        print(f"Eval: loss={val_loss:.4f}, acc={val_acc:.4f}")
        print("Confusion matrix shape:", cm.shape)


if __name__ == '__main__':
    main()
