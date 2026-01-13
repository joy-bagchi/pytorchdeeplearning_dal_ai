import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ----------------------------
# Models
# ----------------------------

class ThreeLayerCNN(nn.Module):
    """
    Matches the lab-style "visualization" CNN: it outputs a feature map (not logits).
    Useful for feature map visualization.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class ModelSpec:
    name: str
    ctor: callable
    weights: Optional[object]  # torchvision weights enum or None
    is_classifier: bool


def available_model_specs() -> List[ModelSpec]:
    # Using torchvision weight metadata gives us class labels + recommended transforms.
    return [
        ModelSpec(
            name="ResNet18 (ImageNet pretrained)",
            ctor=models.resnet18,
            weights=models.ResNet18_Weights.DEFAULT,
            is_classifier=True,
        ),
        ModelSpec(
            name="MobileNetV3-Small (ImageNet pretrained)",
            ctor=models.mobilenet_v3_small,
            weights=models.MobileNet_V3_Small_Weights.DEFAULT,
            is_classifier=True,
        ),
        ModelSpec(
            name="EfficientNet-B0 (ImageNet pretrained)",
            ctor=models.efficientnet_b0,
            weights=models.EfficientNet_B0_Weights.DEFAULT,
            is_classifier=True,
        ),
        ModelSpec(
            name="ThreeLayerCNN (feature maps only, like lab)",
            ctor=ThreeLayerCNN,
            weights=None,
            is_classifier=False,
        ),
    ]


# ----------------------------
# Helpers: preprocessing, hooks, visualizations
# ----------------------------

def pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def overlay_heatmap_on_image(
    base_rgb: np.ndarray, heat_01: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """
    base_rgb: HxWx3 in [0,255]
    heat_01:  HxW   in [0,1]
    """
    heat = (heat_01 * 255).astype(np.uint8)
    # "Jet-like" simple colormap without extra deps:
    # We'll map heat to RGB via a piecewise function.
    h = heat_01
    r = np.clip(1.5 - np.abs(4 * (h - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4 * (h - 0.50)), 0, 1)
    b = np.clip(1.5 - np.abs(4 * (h - 0.25)), 0, 1)
    heat_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    out = (alpha * heat_rgb + (1 - alpha) * base_rgb).astype(np.uint8)
    return out


class ActivationCatcher:
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}

    def hook(self, name: str):
        def _fn(_module, _inp, out):
            self.activations[name] = out.detach()
        return _fn


class GradCamCatcher:
    def __init__(self):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

    def fwd_hook(self, _module, _inp, out):
        self.activations = out

    def bwd_hook(self, _module, _grad_in, grad_out):
        # grad_out is a tuple; first element corresponds to output gradient
        self.gradients = grad_out[0]


def list_named_modules(model: nn.Module) -> List[str]:
    names = []
    for n, m in model.named_modules():
        # skip top-level empty name
        if n.strip() == "":
            continue
        # We mostly want conv blocks or meaningful containers
        names.append(n)
    return names


def find_last_conv_layer_name(model: nn.Module) -> Optional[str]:
    last_name = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name = n
    return last_name


def make_preprocess(weights) -> transforms.Compose:
    # Torchvision weights provide the recommended preprocessing pipeline
    if weights is None:
        # Reasonable default for feature visualization
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    return weights.transforms()


def infer_labels(weights) -> Optional[List[str]]:
    if weights is None:
        return None
    meta = getattr(weights, "meta", None)
    if meta and "categories" in meta:
        return list(meta["categories"])
    return None


def tensor_to_feature_grid(
    feat: torch.Tensor,
    max_maps: int = 32,
    cols: int = 8,
) -> np.ndarray:
    """
    feat: 1xCxHxW
    Returns an image grid (H_grid x W_grid) in uint8 (grayscale).
    """
    assert feat.ndim == 4 and feat.shape[0] == 1
    c = feat.shape[1]
    n = min(c, max_maps)
    cols = min(cols, n)
    rows = int(math.ceil(n / cols))

    # Normalize each map independently for display
    maps = feat[0, :n].cpu().numpy()
    maps_norm = []
    for i in range(n):
        maps_norm.append(normalize_01(maps[i]))
    maps_norm = np.stack(maps_norm, axis=0)  # NxHxW

    H, W = maps_norm.shape[1], maps_norm.shape[2]
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)

    for i in range(n):
        r = i // cols
        col = i % cols
        grid[r * H : (r + 1) * H, col * W : (col + 1) * W] = maps_norm[i]

    return (grid * 255).astype(np.uint8)


def compute_saliency(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Vanilla saliency: |d score / d input|
    Returns HxW float in [0,1]
    """
    model.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)

    out = model(x)
    if out.ndim != 2:
        raise ValueError("Saliency requires classifier output logits [1, num_classes].")

    if class_idx is None:
        class_idx = int(out.argmax(dim=1).item())

    score = out[0, class_idx]
    score.backward()

    grad = x.grad.detach()[0]  # 3xHxW
    sal = grad.abs().max(dim=0).values.cpu().numpy()  # HxW
    return normalize_01(sal)


def compute_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer_name: str,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Grad-CAM for a given layer name (should be conv-ish)
    Returns HxW float in [0,1] (upsampled to input size)
    """
    model.zero_grad(set_to_none=True)

    # Locate layer
    layer = dict(model.named_modules()).get(target_layer_name, None)
    if layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in model.")

    catcher = GradCamCatcher()
    h1 = layer.register_forward_hook(catcher.fwd_hook)

    # full_backward_hook is safer for modern PyTorch
    try:
        h2 = layer.register_full_backward_hook(catcher.bwd_hook)
    except Exception:
        h2 = layer.register_backward_hook(catcher.bwd_hook)

    out = model(x)

    if out.ndim != 2:
        h1.remove(); h2.remove()
        raise ValueError("Grad-CAM requires classifier output logits [1, num_classes].")

    if class_idx is None:
        class_idx = int(out.argmax(dim=1).item())

    score = out[0, class_idx]
    score.backward()

    acts = catcher.activations  # 1xCxHxW
    grads = catcher.gradients   # 1xCxHxW
    h1.remove(); h2.remove()

    if acts is None or grads is None:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

    # weights: global-average-pool gradients over spatial dims
    w = grads.mean(dim=(2, 3), keepdim=True)  # 1xCx1x1
    cam = (w * acts).sum(dim=1, keepdim=True)  # 1x1xHxW
    cam = F.relu(cam)

    # Normalize and upsample to input size
    cam = cam.detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_up = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
    heat = cam_up[0, 0].cpu().numpy()
    return normalize_01(heat)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CNN Decision Visualizer", layout="wide")
st.title("CNN Decision Visualizer (PyTorch)")
st.caption("Upload an image, run a CNN, and visualize *why* it predicts what it predicts.")


with st.sidebar:
    st.header("Model")
    specs = available_model_specs()
    spec_name = st.selectbox("Choose a model", [s.name for s in specs])
    spec = next(s for s in specs if s.name == spec_name)

    device = get_device()
    st.write(f"Device: **{device.type}**")

    st.divider()
    st.header("Image")
    uploaded = st.file_uploader("Upload an image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    topk = st.slider("Top-K predictions", min_value=1, max_value=10, value=5)
    st.divider()
    st.header("Visualizations")
    show_saliency = st.checkbox("Saliency map (input gradients)", value=True, disabled=not spec.is_classifier)
    show_gradcam = st.checkbox("Grad-CAM heatmap", value=True, disabled=not spec.is_classifier)
    show_featuremaps = st.checkbox("Feature maps / activations", value=True)

    gradcam_alpha = st.slider("Grad-CAM overlay strength", 0.0, 0.9, 0.45, 0.05) if spec.is_classifier else 0.45
    max_feature_maps = st.slider("Max feature maps to show", 8, 64, 32, 8) if show_featuremaps else 32
    feature_cols = st.slider("Feature map grid columns", 2, 12, 8, 1) if show_featuremaps else 8


@st.cache_resource
def load_model(_spec: ModelSpec) -> Tuple[nn.Module, transforms.Compose, Optional[List[str]]]:
    if _spec.weights is None:
        model = _spec.ctor()
        preprocess = make_preprocess(None)
        labels = None
    else:
        model = _spec.ctor(weights=_spec.weights)
        preprocess = make_preprocess(_spec.weights)
        labels = infer_labels(_spec.weights)

    model.eval()
    return model, preprocess, labels


if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()


# Load + show the image
img = Image.open(io.BytesIO(uploaded.read()))
img = pil_to_rgb(img)

colA, colB = st.columns([1, 2], vertical_alignment="top")
with colA:
    st.subheader("Input")
    st.image(img, use_container_width=True)

model, preprocess, labels = load_model(spec)
model = model.to(device)

# Preprocess
x = preprocess(img).unsqueeze(0).to(device)

# Run inference
with torch.inference_mode():
    out = model(x)

# If classifier: compute probs and top-k
if spec.is_classifier:
    logits = out
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    top_idx = probs.argsort()[::-1][:topk]
    top_items = []
    for i in top_idx:
        name = labels[i] if labels is not None else str(i)
        top_items.append((int(i), name, float(probs[i])))

    with colB:
        st.subheader("Prediction")
        st.write("Top predictions:")
        st.dataframe(
            {
                "class_id": [t[0] for t in top_items],
                "label": [t[1] for t in top_items],
                "probability": [t[2] for t in top_items],
            },
            use_container_width=True,
            hide_index=True,
        )

    selected_class = st.selectbox(
        "Choose class for saliency/Grad-CAM (defaults to top-1)",
        options=[(t[0], t[1]) for t in top_items],
        format_func=lambda z: f"{z[1]} (id={z[0]})",
    )
    class_idx = int(selected_class[0])

else:
    with colB:
        st.subheader("Output")
        st.write(
            "This model outputs **feature maps** (not class logits). "
            "Use the Feature Maps section to see what each layer detects."
        )
    class_idx = None


st.divider()
tabs = st.tabs(["Saliency", "Grad-CAM", "Feature Maps"])


# ----------------------------
# Saliency
# ----------------------------
with tabs[0]:
    if not spec.is_classifier:
        st.warning("Saliency requires a classifier (logits output). Choose a pretrained ImageNet model.")
    elif not show_saliency:
        st.info("Enable Saliency in the sidebar to display it.")
    else:
        try:
            sal = compute_saliency(model, x, class_idx=class_idx)
            base = np.array(img.resize((sal.shape[1], sal.shape[0])))
            sal_rgb = overlay_heatmap_on_image(base, sal, alpha=0.55)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Saliency heatmap (overlay)**")
                st.image(sal_rgb, use_container_width=True)
            with c2:
                st.markdown("**Saliency (raw)**")
                st.image((sal * 255).astype(np.uint8), clamp=True, use_container_width=True)
        except Exception as e:
            st.error(f"Saliency failed: {e}")


# ----------------------------
# Grad-CAM
# ----------------------------
with tabs[1]:
    if not spec.is_classifier:
        st.warning("Grad-CAM requires a classifier (logits output). Choose a pretrained ImageNet model.")
    elif not show_gradcam:
        st.info("Enable Grad-CAM in the sidebar to display it.")
    else:
        names = list_named_modules(model)
        default_layer = find_last_conv_layer_name(model) or (names[-1] if names else "")
        layer_name = st.selectbox(
            "Choose a layer for Grad-CAM (usually the last conv layer works best)",
            options=names,
            index=names.index(default_layer) if default_layer in names else 0,
        )

        try:
            heat = compute_gradcam(model, x, target_layer_name=layer_name, class_idx=class_idx)
            base = np.array(img.resize((heat.shape[1], heat.shape[0])))
            overlay = overlay_heatmap_on_image(base, heat, alpha=gradcam_alpha)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Grad-CAM (overlay)**")
                st.image(overlay, use_container_width=True)
            with c2:
                st.markdown("**Grad-CAM (raw)**")
                st.image((heat * 255).astype(np.uint8), clamp=True, use_container_width=True)

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")


# ----------------------------
# Feature maps (activations)
# ----------------------------
with tabs[2]:
    if not show_featuremaps:
        st.info("Enable Feature maps in the sidebar to display them.")
    else:
        catcher = ActivationCatcher()
        hooks = []

        # Hook conv layers (or the 3 lab layers) for cleaner visuals
        hook_candidates = []
        if isinstance(model, ThreeLayerCNN):
            hook_candidates = [("layer1", model.layers[0]), ("layer2", model.layers[1]), ("layer3", model.layers[2])]
        else:
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    hook_candidates.append((n, m))

        if len(hook_candidates) == 0:
            st.warning("No Conv2d layers found to visualize.")
        else:
            layer_names = [n for n, _ in hook_candidates]
            chosen = st.selectbox("Choose layer to visualize feature maps", options=layer_names, index=len(layer_names)-1)

            # Register hooks only for chosen layer for speed
            chosen_module = dict(hook_candidates)[chosen]
            hooks.append(chosen_module.register_forward_hook(catcher.hook(chosen)))

            with torch.inference_mode():
                _ = model(x)

            for h in hooks:
                h.remove()

            act = catcher.activations.get(chosen, None)
            if act is None:
                st.error("Failed to capture activations.")
            elif act.ndim != 4:
                st.error(f"Expected 4D activation (1xCxHxW), got shape {tuple(act.shape)}")
            else:
                grid_u8 = tensor_to_feature_grid(act, max_maps=max_feature_maps, cols=feature_cols)
                st.markdown(f"**Layer:** `{chosen}` | activation shape: `{tuple(act.shape)}`")
                st.image(grid_u8, clamp=True, use_container_width=True)

                st.caption(
                    "Each tile is one filter's activation map (normalized independently). "
                    "Early layers often detect edges/textures; later layers become more object-part-like."
                )
