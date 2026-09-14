"""Physics-guided losses for soil-moisture estimation."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_softplus(
    value: float,
) -> float:
    return math.log(
        math.exp(value) - 1.0
    )


class SARObservationPhysics(nn.Module):
    """
    WCM-inspired differentiable observation model.

    This is intentionally a physics-guided proxy,
    NOT a calibrated Water Cloud Model.

    Inputs:
        predicted SSM [% saturation]
        local incidence angle [degrees]
        optional vegetation proxy [0,1]

    Output:
        predicted VV linear backscatter power
    """

    def __init__(self):
        super().__init__()

        self.raw_attenuation = nn.Parameter(
            torch.tensor(
                _inverse_softplus(0.5),
                dtype=torch.float32,
            )
        )

        self.raw_vegetation_scatter = nn.Parameter(
            torch.tensor(
                _inverse_softplus(0.05),
                dtype=torch.float32,
            )
        )

        self.raw_soil_intercept = nn.Parameter(
            torch.tensor(
                _inverse_softplus(0.02),
                dtype=torch.float32,
            )
        )

        self.raw_soil_sensitivity = nn.Parameter(
            torch.tensor(
                _inverse_softplus(0.20),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        ssm_percent: torch.Tensor,
        incidence_angle_deg: torch.Tensor,
        vegetation: torch.Tensor | None = None,
    ) -> torch.Tensor:

        ssm_fraction = (
            ssm_percent / 100.0
        )

        theta = torch.deg2rad(
            incidence_angle_deg
        )

        cos_theta = torch.clamp(
            torch.cos(theta),
            min=0.1,
        )

        if vegetation is None:
            vegetation = torch.zeros_like(
                ssm_fraction
            )

        vegetation = torch.clamp(
            vegetation,
            min=0.0,
            max=1.0,
        )

        attenuation = F.softplus(
            self.raw_attenuation
        )

        vegetation_scatter = F.softplus(
            self.raw_vegetation_scatter
        )

        soil_intercept = F.softplus(
            self.raw_soil_intercept
        )

        soil_sensitivity = F.softplus(
            self.raw_soil_sensitivity
        )

        # Vegetation attenuation.
        transmission = torch.exp(
            -2.0
            * attenuation
            * vegetation
            / cos_theta
        )

        # Soil backscatter increases monotonically
        # with estimated soil moisture.
        soil_sigma0 = (
            soil_intercept
            + soil_sensitivity
            * ssm_fraction
        )

        vegetation_sigma0 = (
            vegetation_scatter
            * vegetation
            * cos_theta
            * (1.0 - transmission)
        )

        vv_predicted = (
            vegetation_sigma0
            + transmission
            * soil_sigma0
        )

        return vv_predicted


def sar_physics_loss(
    predicted_vv: torch.Tensor,
    observed_vv: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        predicted_vv,
        observed_vv,
    )


def uncertainty_weighted_ssm_loss(
    predicted_ssm: torch.Tensor,
    reference_ssm: torch.Tensor,
    reference_noise: torch.Tensor,
) -> torch.Tensor:
    """
    Weight CLMS supervision according to SSM_NOISE.

    Larger reported uncertainty => smaller loss weight.
    """

    element_loss = F.smooth_l1_loss(
        predicted_ssm,
        reference_ssm,
        reduction="none",
    )

    variance = torch.clamp(
        reference_noise**2,
        min=1.0,
    )

    weights = 1.0 / variance

    # Normalize weights so overall loss scale
    # stays approximately stable.
    weights = weights / torch.clamp(
        weights.mean(),
        min=1e-6,
    )

    return (
        weights * element_loss
    ).mean()