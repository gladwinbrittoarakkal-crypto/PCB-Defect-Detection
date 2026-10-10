import torch, gradio as gr
from pcb_core import build_model, detect, draw
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(pretrained=False)
sd = {k: (v.float() if v.is_floating_point() else v)
      for k, v in torch.load("models/best.pt", map_location="cpu").items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all("num_batches_tracked" in k for k in missing), (missing, unexpected)
model.to(dev).eval()

def run(img, conf):
    dets = detect(model, img, conf, dev)
    return draw(img, dets), [[d["label"], round(d["score"], 3), [round(v) for v in d["box"]]] for d in dets]

demo = gr.Interface(run, [gr.Image(type="pil", label="PCB image"), gr.Slider(0.1, 0.95, 0.5, label="Confidence")],
                    [gr.Image(label="Detections"), gr.Dataframe(headers=["defect", "score", "box"])],
                    title="PCB Defect Detector")
if __name__ == "__main__": demo.launch()
