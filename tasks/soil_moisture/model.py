"""GeoFM + optional temporal JEPA + optional physics for soil moisture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.geofm.terramind import (
    TerraMindConfig,
    TerraMindEncoder,
)
from models.predictive.temporal_jepa import (
    TemporalJEPAPredictor,
    jepa_latent_loss,
)
from models.heads.soil_moisture import (
    SoilMoistureHead,
)
from physics.soil_moisture import (
    SARObservationPhysics,
    sar_physics_loss,
    uncertainty_weighted_ssm_loss,
)


@dataclass
class LossWeights:
    """
    Initial loss weights.

    These are experimental starting values, not calibrated constants.
    """

    retrieval: float = 1.0
    predictive: float = 0.5
    jepa: float = 0.20
    physics: float = 0.02


class GeoFMJEPASoilMoisture(nn.Module):
    """
    TerraMind soil-moisture model with explicit ablation switches.

    Supported configurations
    ------------------------
    1. GeoFM only
        use_jepa=False
        use_physics=False

    2. GeoFM + JEPA
        use_jepa=True
        use_physics=False

    3. GeoFM + physics
        use_jepa=False
        use_physics=True

    4. GeoFM + JEPA + physics
        use_jepa=True
        use_physics=True

    Sentinel-2 remains optional independently of these ablations.
    """

    def __init__(
        self,
        use_s2: bool = False,
        use_jepa: bool = True,
        use_physics: bool = True,
        freeze_geofm: bool = True,
        loss_weights: LossWeights | None = None,
    ):
        super().__init__()

        self.use_s2 = use_s2
        self.use_jepa = use_jepa
        self.use_physics = use_physics
        self.freeze_geofm = freeze_geofm

        self.loss_weights = (
            loss_weights
            if loss_weights is not None
            else LossWeights()
        )

        # -----------------------------------------------------
        # 1. Pretrained GeoFM
        # -----------------------------------------------------

        self.encoder = TerraMindEncoder(
            TerraMindConfig(
                model_name="terramind_v1_base",
                use_s2=use_s2,
                freeze_backbone=freeze_geofm,
                embedding_dim=768,
            )
        )

        embedding_dim = (
            self.encoder.config.embedding_dim
        )

        # -----------------------------------------------------
        # 2. Shared soil-moisture head
        #
        # Create this BEFORE optional modules so that, under
        # a fixed seed, all ablations start from the same
        # SSM-head initialization.
        # -----------------------------------------------------

        self.ssm_head = SoilMoistureHead(
            embedding_dim=embedding_dim,
        )

        # -----------------------------------------------------
        # 3. Optional temporal JEPA
        # -----------------------------------------------------

        if self.use_jepa:
            self.jepa = TemporalJEPAPredictor(
                embedding_dim=embedding_dim,
            )
        else:
            self.jepa = None

        # -----------------------------------------------------
        # 4. Optional differentiable SAR physics
        # -----------------------------------------------------

        if self.use_physics:
            self.physics = SARObservationPhysics()
        else:
            self.physics = None

        # -----------------------------------------------------
        # 5. Device consistency
        # -----------------------------------------------------

        self.device_name = (
            self.encoder.device_name
        )

        self.to(
            self.device_name
        )

    @property
    def configuration_name(self) -> str:
        """
        Human-readable experiment name.
        """

        if (
            not self.use_jepa
            and not self.use_physics
        ):
            return "GeoFM"

        if (
            self.use_jepa
            and not self.use_physics
        ):
            return "GeoFM+JEPA"

        if (
            not self.use_jepa
            and self.use_physics
        ):
            return "GeoFM+Physics"

        return "GeoFM+JEPA+Physics"

    def forward(
        self,
        s1_sequence: torch.Tensor,
        s2_sequence: torch.Tensor | None = None,
        incidence_angle: torch.Tensor | None = None,
        vegetation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        s1_sequence
            [B, T, 2, H, W]

        s2_sequence
            Optional S2 sequence:
            [B, T, C_s2, H, W]

        incidence_angle
            Target-date local incidence angle:
            [B]

        vegetation
            Optional target-date vegetation proxy:
            [B]

        Returns
        -------
        dict
            Always contains:
                retrieval_ssm
                target_latent
                ssm

            JEPA configurations additionally contain:
                predicted_latent
                predictive_ssm

            Physics configurations additionally contain:
                predicted_vv
        """

        if s1_sequence.ndim != 5:
            raise ValueError(
                "Expected s1_sequence [B,T,C,H,W], "
                f"got {tuple(s1_sequence.shape)}."
            )

        n_times = s1_sequence.shape[1]

        if n_times < 1:
            raise ValueError(
                "At least one temporal observation is required."
            )

        if self.use_jepa and n_times < 2:
            raise ValueError(
                "JEPA requires at least two temporal observations."
            )

        if (
            self.use_s2
            and s2_sequence is None
        ):
            raise ValueError(
                "use_s2=True but no s2_sequence was supplied."
            )

        if s2_sequence is not None:
            if (
                s2_sequence.shape[:2]
                != s1_sequence.shape[:2]
            ):
                raise ValueError(
                    "S1 and S2 must have matching "
                    "batch and temporal dimensions."
                )

        # -----------------------------------------------------
        # GeoFM representations
        #
        # [B,T,C,H,W]
        #       ↓
        # [B,T,D]
        # -----------------------------------------------------

        latent_sequence = (
            self.encoder.encode_sequence(
                s1_sequence=s1_sequence,
                s2_sequence=s2_sequence,
            )
        )

        # Actual target-date representation.
        target_latent = (
            latent_sequence[:, -1]
        )

        # -----------------------------------------------------
        # Retrieval branch
        #
        # Actual target-date S1
        #       ↓
        # TerraMind z(t)
        #       ↓
        # SSM
        # -----------------------------------------------------

        retrieval_ssm = self.ssm_head(
            target_latent
        )

        output = {
            # Compatibility alias.
            "ssm": retrieval_ssm,

            "retrieval_ssm": retrieval_ssm,
            "target_latent": target_latent,
        }

        # -----------------------------------------------------
        # Optional JEPA branch
        #
        # z(t-k)...z(t-1)
        #       ↓
        # JEPA
        #       ↓
        # z_hat(t)
        #       ↓
        # predictive SSM
        # -----------------------------------------------------

        if self.use_jepa:

            context_latents = (
                latent_sequence[:, :-1]
            )

            predicted_latent = self.jepa(
                context_latents
            )

            predictive_ssm = self.ssm_head(
                predicted_latent
            )

            output[
                "predicted_latent"
            ] = predicted_latent

            output[
                "predictive_ssm"
            ] = predictive_ssm

        # -----------------------------------------------------
        # Optional SAR physics branch
        #
        # Physics is tied to retrieval SSM because retrieval
        # corresponds to the actual target-date S1 observation.
        # -----------------------------------------------------

        if self.use_physics:

            if incidence_angle is None:
                raise ValueError(
                    "Physics is enabled but "
                    "incidence_angle was not supplied."
                )

            device = retrieval_ssm.device
            dtype = retrieval_ssm.dtype

            incidence_angle = (
                incidence_angle.to(
                    device=device,
                    dtype=dtype,
                )
            )

            if vegetation is not None:
                vegetation = vegetation.to(
                    device=device,
                    dtype=dtype,
                )

            output[
                "predicted_vv"
            ] = self.physics(
                ssm_percent=retrieval_ssm,
                incidence_angle_deg=incidence_angle,
                vegetation=vegetation,
            )

        return output

    def compute_loss(
        self,
        output: dict[str, torch.Tensor],
        reference_ssm: torch.Tensor,
        reference_noise: torch.Tensor,
        observed_vv: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the active ablation objective.

        GeoFM:
            L = L_retrieval

        GeoFM + JEPA:
            L = L_retrieval
              + λ_pred L_predictive
              + λ_jepa L_JEPA

        GeoFM + Physics:
            L = L_retrieval
              + λ_phys L_physics

        GeoFM + JEPA + Physics:
            L = L_retrieval
              + λ_pred L_predictive
              + λ_jepa L_JEPA
              + λ_phys L_physics
        """

        device = (
            output["retrieval_ssm"].device
        )

        dtype = (
            output["retrieval_ssm"].dtype
        )

        reference_ssm = reference_ssm.to(
            device=device,
            dtype=dtype,
        )

        reference_noise = (
            reference_noise.to(
                device=device,
                dtype=dtype,
            )
        )

        # -----------------------------------------------------
        # Retrieval supervision: always active.
        # -----------------------------------------------------

        loss_retrieval = (
            uncertainty_weighted_ssm_loss(
                predicted_ssm=output[
                    "retrieval_ssm"
                ],
                reference_ssm=reference_ssm,
                reference_noise=reference_noise,
            )
        )

        zero = torch.zeros(
            (),
            device=device,
            dtype=dtype,
        )

        loss_predictive = zero
        loss_jepa = zero
        loss_physics = zero

        # -----------------------------------------------------
        # JEPA losses
        # -----------------------------------------------------

        if self.use_jepa:

            loss_predictive = (
                uncertainty_weighted_ssm_loss(
                    predicted_ssm=output[
                        "predictive_ssm"
                    ],
                    reference_ssm=reference_ssm,
                    reference_noise=reference_noise,
                )
            )

            loss_jepa = jepa_latent_loss(
                predicted=output[
                    "predicted_latent"
                ],
                target=output[
                    "target_latent"
                ],
            )

        # -----------------------------------------------------
        # Physics loss
        # -----------------------------------------------------

        if self.use_physics:

            if observed_vv is None:
                raise ValueError(
                    "Physics is enabled but observed_vv "
                    "was not supplied."
                )

            if "predicted_vv" not in output:
                raise RuntimeError(
                    "Physics is enabled but predicted_vv "
                    "is missing from model output."
                )

            observed_vv = observed_vv.to(
                device=device,
                dtype=dtype,
            )

            loss_physics = sar_physics_loss(
                predicted_vv=output[
                    "predicted_vv"
                ],
                observed_vv=observed_vv,
            )

        # -----------------------------------------------------
        # Combined active objective
        # -----------------------------------------------------

        total = (
            self.loss_weights.retrieval
            * loss_retrieval
        )

        if self.use_jepa:
            total = (
                total
                + self.loss_weights.predictive
                * loss_predictive
                + self.loss_weights.jepa
                * loss_jepa
            )

        if self.use_physics:
            total = (
                total
                + self.loss_weights.physics
                * loss_physics
            )

        return {
            "total": total,
            "retrieval": loss_retrieval,
            "predictive": loss_predictive,
            "jepa": loss_jepa,
            "physics": loss_physics,
        }

#"""GeoFM + temporal JEPA + physics model for soil-moisture estimation."""
#
#from __future__ import annotations
#
#from dataclasses import dataclass
#
#import torch
#from torch import nn
#
#from models.geofm.terramind import (
#    TerraMindConfig,
#    TerraMindEncoder,
#)
#from models.predictive.temporal_jepa import (
#    TemporalJEPAPredictor,
#    jepa_latent_loss,
#)
#from models.heads.soil_moisture import SoilMoistureHead
#from physics.soil_moisture import (
#    SARObservationPhysics,
#    sar_physics_loss,
#    uncertainty_weighted_ssm_loss,
#)
#
#
#@dataclass
#class LossWeights:
#    """
#    Relative weights for the multi-objective training loss.
#
#    These are initial experimental values, not calibrated constants.
#    """
#
#    retrieval: float = 1.0
#    predictive: float = 0.5
#    jepa: float = 0.20
#    physics: float = 0.02
#
#
#class GeoFMJEPASoilMoisture(nn.Module):
#    """
#    Soil-moisture model composed of:
#
#        S1 (+ optional S2)
#            ↓
#        TerraMind GeoFM
#            ↓
#        temporal latent sequence z1 ... zT
#            ├───────────────┐
#            │               │
#            ▼               ▼
#        target zT       JEPA(z1...zT-1)
#            │               │
#            │               ▼
#            │            z_hat_T
#            │               │
#            ▼               ▼
#        retrieval SSM   predictive SSM
#            │
#            ▼
#        SAR physics consistency
#
#    Retrieval SSM:
#        estimated from the actual target-date TerraMind representation.
#
#    Predictive SSM:
#        estimated from the JEPA-predicted target-date representation.
#
#    This separation prevents the model from hiding the JEPA contribution
#    inside an opaque latent fusion mechanism.
#    """
#
#    def __init__(
#        self,
#        use_s2: bool = False,
#        freeze_geofm: bool = True,
#        loss_weights: LossWeights | None = None,
#    ):
#        super().__init__()
#
#        self.use_s2 = use_s2
#        self.freeze_geofm = freeze_geofm
#
#        self.loss_weights = (
#            loss_weights
#            if loss_weights is not None
#            else LossWeights()
#        )
#
#        # -------------------------------------------------------------
#        # 1. Pretrained GeoFM
#        # -------------------------------------------------------------
#
#        self.encoder = TerraMindEncoder(
#            TerraMindConfig(
#                model_name="terramind_v1_base",
#                use_s2=use_s2,
#                freeze_backbone=freeze_geofm,
#                embedding_dim=768,
#            )
#        )
#
#        embedding_dim = self.encoder.config.embedding_dim
#
#        # -------------------------------------------------------------
#        # 2. Temporal JEPA predictor
#        # -------------------------------------------------------------
#
#        self.jepa = TemporalJEPAPredictor(
#            embedding_dim=embedding_dim,
#        )
#
#        # -------------------------------------------------------------
#        # 3. Soil-moisture regression head
#        #
#        # Shared by retrieval and predictive representations.
#        # -------------------------------------------------------------
#
#        self.ssm_head = SoilMoistureHead(
#            embedding_dim=embedding_dim,
#        )
#
#        # -------------------------------------------------------------
#        # 4. Differentiable SAR observation constraint
#        # -------------------------------------------------------------
#
#        self.physics = SARObservationPhysics()
#
#        # -------------------------------------------------------------
#        # 5. Put all components on the same device.
#        # -------------------------------------------------------------
#
#        self.device_name = self.encoder.device_name
#        self.to(self.device_name)
#
#    def forward(
#        self,
#        s1_sequence: torch.Tensor,
#        s2_sequence: torch.Tensor | None = None,
#        incidence_angle: torch.Tensor | None = None,
#        vegetation: torch.Tensor | None = None,
#    ) -> dict[str, torch.Tensor]:
#        """
#        Forward pass.
#
#        Parameters
#        ----------
#        s1_sequence
#            Shape [B, T, 2, H, W].
#
#        s2_sequence
#            Optional S2 sequence.
#            Shape [B, T, C_s2, H, W].
#
#        incidence_angle
#            Target-date local incidence angle.
#            Shape [B].
#
#        vegetation
#            Optional vegetation proxy in [0, 1].
#            Shape [B].
#
#        Returns
#        -------
#        dict
#            retrieval_ssm:
#                Soil moisture estimated from actual target representation.
#
#            predictive_ssm:
#                Soil moisture estimated from JEPA-predicted representation.
#
#            target_latent:
#                Actual target-date TerraMind representation.
#
#            predicted_latent:
#                JEPA prediction of target-date representation.
#
#            predicted_vv:
#                Optional physics-model VV estimate.
#        """
#
#        if s1_sequence.ndim != 5:
#            raise ValueError(
#                "Expected s1_sequence with shape "
#                "[B,T,C,H,W], "
#                f"got {tuple(s1_sequence.shape)}."
#            )
#
#        if s1_sequence.shape[1] < 2:
#            raise ValueError(
#                "Temporal JEPA requires at least two observations."
#            )
#
#        if self.use_s2 and s2_sequence is None:
#            raise ValueError(
#                "Model was configured with use_s2=True "
#                "but no s2_sequence was provided."
#            )
#
#        if (
#            s2_sequence is not None
#            and s2_sequence.shape[:2]
#            != s1_sequence.shape[:2]
#        ):
#            raise ValueError(
#                "S1 and S2 sequences must have matching "
#                "batch and temporal dimensions."
#            )
#
#        # -------------------------------------------------------------
#        # GeoFM temporal representations
#        #
#        # [B,T,C,H,W] -> [B,T,D]
#        # -------------------------------------------------------------
#
#        latent_sequence = self.encoder.encode_sequence(
#            s1_sequence=s1_sequence,
#            s2_sequence=s2_sequence,
#        )
#
#        # Context = all observations before the target date.
#        context_latents = latent_sequence[:, :-1]
#
#        # Actual target representation.
#        target_latent = latent_sequence[:, -1]
#
#        # -------------------------------------------------------------
#        # JEPA: predict target latent from previous representations.
#        # -------------------------------------------------------------
#
#        predicted_latent = self.jepa(
#            context_latents
#        )
#
#        # -------------------------------------------------------------
#        # Soil-moisture retrieval from actual observation.
#        # -------------------------------------------------------------
#
#        retrieval_ssm = self.ssm_head(
#            target_latent
#        )
#
#        # -------------------------------------------------------------
#        # Soil-moisture prediction from JEPA representation.
#        # -------------------------------------------------------------
#
#        predictive_ssm = self.ssm_head(
#            predicted_latent
#        )
#
#        output = {
#            # Alias retained for compatibility with existing code.
#            "ssm": retrieval_ssm,
#
#            "retrieval_ssm": retrieval_ssm,
#            "predictive_ssm": predictive_ssm,
#
#            "target_latent": target_latent,
#            "predicted_latent": predicted_latent,
#        }
#
#        # -------------------------------------------------------------
#        # Optional SAR observation-physics constraint.
#        #
#        # Physics is tied to retrieval SSM because that estimate
#        # corresponds to the actual target-date S1 observation.
#        # -------------------------------------------------------------
#
#        if incidence_angle is not None:
#            device = retrieval_ssm.device
#            dtype = retrieval_ssm.dtype
#
#            incidence_angle = incidence_angle.to(
#                device=device,
#                dtype=dtype,
#            )
#
#            if vegetation is not None:
#                vegetation = vegetation.to(
#                    device=device,
#                    dtype=dtype,
#                )
#
#            output["predicted_vv"] = self.physics(
#                ssm_percent=retrieval_ssm,
#                incidence_angle_deg=incidence_angle,
#                vegetation=vegetation,
#            )
#
#        return output
#
#    def compute_loss(
#        self,
#        output: dict[str, torch.Tensor],
#        reference_ssm: torch.Tensor,
#        reference_noise: torch.Tensor,
#        observed_vv: torch.Tensor | None = None,
#    ) -> dict[str, torch.Tensor]:
#        """
#        Compute retrieval + prediction + JEPA + physics losses.
#
#        L_total =
#            λ_retrieval  * L_retrieval
#          + λ_predictive * L_predictive
#          + λ_jepa       * L_jepa
#          + λ_physics    * L_physics
#        """
#
#        device = output["retrieval_ssm"].device
#        dtype = output["retrieval_ssm"].dtype
#
#        reference_ssm = reference_ssm.to(
#            device=device,
#            dtype=dtype,
#        )
#
#        reference_noise = reference_noise.to(
#            device=device,
#            dtype=dtype,
#        )
#
#        if observed_vv is not None:
#            observed_vv = observed_vv.to(
#                device=device,
#                dtype=dtype,
#            )
#
#        # -------------------------------------------------------------
#        # 1. Retrieval supervision
#        #
#        # Actual target-date GeoFM representation -> SSM
#        # -------------------------------------------------------------
#
#        loss_retrieval = uncertainty_weighted_ssm_loss(
#            predicted_ssm=output["retrieval_ssm"],
#            reference_ssm=reference_ssm,
#            reference_noise=reference_noise,
#        )
#
#        # -------------------------------------------------------------
#        # 2. Predictive SSM supervision
#        #
#        # JEPA-predicted target representation -> SSM
#        # -------------------------------------------------------------
#
#        loss_predictive = uncertainty_weighted_ssm_loss(
#            predicted_ssm=output["predictive_ssm"],
#            reference_ssm=reference_ssm,
#            reference_noise=reference_noise,
#        )
#
#        # -------------------------------------------------------------
#        # 3. JEPA latent prediction
#        # -------------------------------------------------------------
#
#        loss_jepa = jepa_latent_loss(
#            predicted=output["predicted_latent"],
#            target=output["target_latent"],
#        )
#
#        # -------------------------------------------------------------
#        # 4. SAR physics consistency
#        # -------------------------------------------------------------
#
#        loss_physics = torch.zeros(
#            (),
#            device=device,
#            dtype=dtype,
#        )
#
#        if (
#            observed_vv is not None
#            and "predicted_vv" in output
#        ):
#            loss_physics = sar_physics_loss(
#                predicted_vv=output["predicted_vv"],
#                observed_vv=observed_vv,
#            )
#
#        # -------------------------------------------------------------
#        # Combined objective
#        # -------------------------------------------------------------
#
#        total = (
#            self.loss_weights.retrieval
#            * loss_retrieval
#
#            + self.loss_weights.predictive
#            * loss_predictive
#
#            + self.loss_weights.jepa
#            * loss_jepa
#
#            + self.loss_weights.physics
#            * loss_physics
#        )
#
#        return {
#            "total": total,
#            "retrieval": loss_retrieval,
#            "predictive": loss_predictive,
#            "jepa": loss_jepa,
#            "physics": loss_physics,
#        }