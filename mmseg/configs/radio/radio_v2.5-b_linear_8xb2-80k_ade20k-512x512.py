# fmt: off
import os

_base_ = ["./radio_linear_8xb2-80k_ade20k-512x512.py"]

model = dict(
    backbone=dict(
        repo_id="nvidia/RADIO-B",
    ),
)