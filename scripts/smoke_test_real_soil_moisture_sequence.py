"""Real-data smoke test for GeoFM + JEPA + physics."""

from __future__ import annotations

import torch

from tasks.soil_moisture.dataset import (
    materialize_timeseries,
)

from tasks.soil_moisture.preprocessing import (
    prepare_s1_terramind,
)

from tasks.soil_moisture.model import (
    GeoFMJEPASoilMoisture,
)


BBOX = [
    10.00,
    45.05,
    10.20,
    45.20,
]


records = materialize_timeseries(
    bbox=BBOX,
    start_date="2026-05-01",
    end_date="2026-08-31",
    orbit_state="descending",
    relative_orbit=168,
    max_scenes=3,
    width=224,
    height=224,
)


if len(records) < 3:
    raise RuntimeError(
        "Need at least three same-orbit "
        "Sentinel-1 observations."
    )


print("\nTemporal sequence:")

for record in records:
    print(
        record["date"],
        record["relative_orbit"],
        record["scene_id"],
    )


# ---------------------------------------------------------
# Prepare real S1 GeoFM inputs.
# ---------------------------------------------------------

frames = []

for record in records:

    frame = prepare_s1_terramind(
        record["s1_path"],
        image_size=224,
    )

    if not torch.isfinite(
        frame
    ).all():
        raise RuntimeError(
            f"Non-finite S1 input "
            f"for {record['date']}."
        )

    frames.append(
        frame
    )


# [T,C,H,W]
sequence = torch.stack(
    frames,
    dim=0,
)

# [B,T,C,H,W]
sequence = sequence.unsqueeze(
    0
)

print(
    "\nS1 sequence:",
    sequence.shape,
)


# ---------------------------------------------------------
# Target information comes from final date.
# ---------------------------------------------------------

target = records[-1]

s1_stats = target[
    "sentinel1"
]

reference_stats = target[
    "reference"
]


incidence_angle = torch.tensor(
    [
        s1_stats[
            "incidence_angle"
        ]["median"]
    ],
    dtype=torch.float32,
)


observed_vv = torch.tensor(
    [
        s1_stats[
            "vv"
        ]["median"]
    ],
    dtype=torch.float32,
)


reference_ssm = torch.tensor(
    [
        reference_stats[
            "ssm_percent_saturation"
        ]["median"]
    ],
    dtype=torch.float32,
)


reference_noise = torch.tensor(
    [
        reference_stats[
            "ssm_noise"
        ]["median"]
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = GeoFMJEPASoilMoisture(
    use_s2=False,
    freeze_geofm=True,
)

model.train()


output = model(
    s1_sequence=sequence,
    incidence_angle=incidence_angle,
)


losses = model.compute_loss(
    output=output,
    reference_ssm=reference_ssm,
    reference_noise=reference_noise,
    observed_vv=observed_vv,
)


print("\nTarget date:")
print(
    target["date"]
)

print(
    "Reference SSM:",
    reference_ssm.item(),
)

print(
    "Reference noise:",
    reference_noise.item(),
)

print(
    "Observed VV median:",
    observed_vv.item(),
)

print(
    "Incidence angle:",
    incidence_angle.item(),
)


print("\nUntrained model outputs:")

print(
    "Retrieval SSM:",
    output[
        "retrieval_ssm"
    ].detach().item(),
)

print(
    "Predictive SSM:",
    output[
        "predictive_ssm"
    ].detach().item(),
)

print(
    "Predicted VV:",
    output[
        "predicted_vv"
    ].detach().item(),
)


print("\nLosses:")

for name, loss in losses.items():
    print(
        f"{name:12s}",
        float(
            loss.detach()
        ),
        "finite=",
        bool(
            torch.isfinite(
                loss
            )
        ),
    )


losses[
    "total"
].backward()


print(
    "\nBackward: OK"
)

print(
    "JEPA gradient:",
    next(
        model.jepa.parameters()
    ).grad is not None,
)

print(
    "SSM gradient:",
    next(
        model.ssm_head.parameters()
    ).grad is not None,
)

print(
    "Physics gradient:",
    next(
        model.physics.parameters()
    ).grad is not None,
)

print(
    "TerraMind gradient:",
    any(
        p.grad is not None
        for p in model.encoder.backbone.parameters()
    ),
)