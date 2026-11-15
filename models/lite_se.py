"""
LiteSE — lightweight Squeeze-and-Excitation channel attention for YOLOv8s-Lite-SE.

    input  [B, C, H, W]
      |
      +---------------------------------------+
      |                                       |
      |  GlobalAvgPool (AdaptiveAvgPool2d(1)) |
      |              [B, C, 1, 1]             |
      |  Conv2d 1x1  C -> C//reduction        |
      |  ReLU                                 |
      |  Conv2d 1x1  C//reduction -> C        |
      |  Sigmoid                              |
      |              [B, C, 1, 1]             |
      +--------- elementwise multiply --------+
      |
    output [B, C, H, W]   (shape preserved)

Ops used (all ONNX-standard and QAIRT/QNN-friendly):
  GlobalAveragePool (AdaptiveAvgPool2d), Conv2d 1x1, ReLU, Sigmoid, Mul.
No BatchNorm, no transformer attention, no custom/inplace operators.

Usage:
    import ultralytics.nn.tasks as tasks
    from lite_se import LiteSE
    tasks.LiteSE = LiteSE   # parse_model() resolves YAML module names via tasks' globals
    model = YOLO("yolov8s_lite_se.yaml")
"""
import torch.nn as nn


class LiteSE(nn.Module):
    """Lightweight channel attention (Squeeze-and-Excitation, reduction 8).

    Accepts both call patterns produced by Ultralytics parse_model():
      - generic-module path (fresh install): LiteSE(768, 8)
      - base-module path (if registered)   : LiteSE(c1, c2, 8)
    """

    def __init__(self, c1: int, reduction: int = 8, *extra):
        if extra:
            # parse_model injected a c2 channel argument (c1, c2, reduction).
            # c2 == c1 here because width_multiple=1.0 and channel counts are
            # explicit in the YAML, so only the trailing reduction is needed.
            reduction = extra[0]
        super().__init__()
        c_mid = max(4, c1 // reduction)  # 768 -> 96
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_mid, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),                      # inplace=False: cleaner ONNX graph
            nn.Conv2d(c_mid, c1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def register_litese(verbose: bool = True) -> None:
    """Make 'LiteSE' resolvable from a YOLO model YAML (Ultralytics 8.4.173).

    In 8.4.173, tasks.parse_model() looks up module names with globals()[m],
    i.e. the namespace of ultralytics.nn.tasks. Adding the class there is
    sufficient for YAML instantiation and training.
    """
    import ultralytics.nn.tasks as tasks
    if not hasattr(tasks, "LiteSE"):
        setattr(tasks, "LiteSE", LiteSE)
        if verbose:
            print("✅ Registered LiteSE in ultralytics.nn.tasks")
    else:
        if verbose:
            print("LiteSE already registered")
