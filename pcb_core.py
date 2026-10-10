import torch, random
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

CLASSES = ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"]
NUM_CLASSES = len(CLASSES) + 1  # + background

def build_model(pretrained=True):
    ag = AnchorGenerator(sizes=((16,),(32,),(64,),(128,),(256,)), aspect_ratios=((0.5,1.0,2.0),)*5)
    w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    m = fasterrcnn_resnet50_fpn(weights=w, weights_backbone=None, min_size=1024, max_size=1707,
                                rpn_anchor_generator=ag)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
    return m

class PCBDataset(torch.utils.data.Dataset):
    def __init__(self, df, train=False):
        self.g = dict(tuple(df.groupby("stem"))); self.stems = list(self.g); self.train = train
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        g = self.g[self.stems[i]]
        img = TF.to_tensor(Image.open(g.img.iloc[0]).convert("RGB"))
        boxes = torch.tensor(g[["x1","y1","x2","y2"]].values, dtype=torch.float32)
        labels = torch.tensor([CLASSES.index(c)+1 for c in g.cls], dtype=torch.int64)
        if self.train and random.random() < 0.5:
            img = img.flip(-1); w = img.shape[-1]; boxes[:, [0,2]] = w - boxes[:, [2,0]]
        return img, {"boxes": boxes, "labels": labels}

def collate(b): return tuple(zip(*b))

@torch.no_grad()
def detect(model, pil_img, conf=0.5, device="cpu"):
    model.eval()
    out = model([TF.to_tensor(pil_img.convert("RGB")).to(device)])[0]
    return [dict(label=CLASSES[l-1], score=float(s), box=[float(v) for v in b])
            for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()) if s >= conf]

def draw(pil_img, dets):
    im = pil_img.convert("RGB").copy(); d = ImageDraw.Draw(im)
    pal = ["red","orange","yellow","lime","cyan","magenta"]
    for x in dets:
        c = pal[CLASSES.index(x["label"])]
        d.rectangle(x["box"], outline=c, width=4)
        d.text((x["box"][0]+3, max(0, x["box"][1]-14)), f'{x["label"]} {x["score"]:.2f}', fill=c)
    return im
