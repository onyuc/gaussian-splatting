# FreeTimeGS (Unofficial)
# Copyright (C) 2025 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.


from math import isqrt
from typing import Optional

import gsplat
import torch
from torch import Tensor, nn


class DynamicGaussians(nn.Module):
    means: nn.Parameter  # (n, 3)
    scales: nn.Parameter  # (n, 3)
    quats: nn.Parameter  # (n, 4)
    opacities: nn.Parameter  # (n, 1)
    sh_0: nn.Parameter  # (n, 1, 3)
    sh_n: nn.Parameter  # (n, (sh_degree + 1) ** 2 - 1, 3)

    times: nn.Parameter  # (n, 1)
    durations: nn.Parameter  # (n, 1)
    velocities: nn.Parameter  # (n, 3)

    sh_degree: int

    def __init__(
        self,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        opacities: Tensor,
        sh_0: Tensor,
        sh_n: Tensor,
        times: Tensor,
        durations: Tensor,
        velocities: Tensor,
    ):
        super().__init__()

        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.opacities = nn.Parameter(opacities.float())
        self.sh_0 = nn.Parameter(sh_0.float())
        self.sh_n = nn.Parameter(sh_n.float())
        self.times = nn.Parameter(times.float())
        self.durations = nn.Parameter(durations.float())
        self.velocities = nn.Parameter(velocities.float())

        self.sh_degree = isqrt(sh_n.shape[1] + 1) - 1

    def forward(
        self,
        t: float | Tensor,
        w2c: Tensor,
        intrinsic: Tensor,
        shape: tuple[int, int],
        clamp: bool = True,
        sh_degree: Optional[int] = None,
    ):
        means_t = self.means + (t - self.times) * self.velocities
        scales = torch.exp(self.scales)

        if sh_degree is None:
            sh_degree = self.sh_degree

        image, alpha, meta = gsplat.rasterization(
            means=means_t,
            quats=self.quats,
            scales=scales,
            opacities=self.temporal_opacity(t).squeeze(-1),
            colors=torch.cat([self.sh_0, self.sh_n], dim=1),
            viewmats=w2c,
            Ks=intrinsic,
            sh_degree=sh_degree,
            width=shape[1],
            height=shape[0],
        )
        if clamp:
            image = image.clone().clamp(0, 1)

        return image, alpha, meta

    def temporal_opacity(self, t: float | Tensor) -> Tensor:
        return torch.sigmoid(self.opacities) * torch.exp(
            -0.5 * ((t - self.times) / self.durations.exp()) ** 2
        )
    def temporal_opacity_logit(self, t: float | Tensor) -> Tensor:
        """Returns opacity in logit space (before sigmoid) at time t"""
        temporal_opacity_sigmoid = self.temporal_opacity(t)
        # Apply logit (inverse sigmoid): logit(y) = log(y / (1 - y))
        # Clamp to avoid log(0) or log(negative)
        temporal_opacity_clamped = temporal_opacity_sigmoid.clamp(1e-8, 1 - 1e-8)
        return torch.logit(temporal_opacity_clamped) 

    # custom code
    def temporal_means(self, t: float | Tensor) -> Tensor:
        return self.means + (t - self.times) * self.velocities

    @classmethod
    def empty(cls, sh_degree: int):
        return cls(
            means=torch.empty((0, 3)),
            scales=torch.empty((0, 3)),
            quats=torch.empty((0, 4)),
            opacities=torch.empty((0, 1)),
            sh_0=torch.empty((0, 1, 3)),
            sh_n=torch.empty((0, (sh_degree + 1) ** 2 - 1, 3)),
            times=torch.empty((0, 1)),
            durations=torch.empty((0, 1)),
            velocities=torch.empty((0, 3)),
        )

    def __or__(self, other):
        return self.__class__(
            means=torch.cat([self.means, other.means]),
            scales=torch.cat([self.scales, other.scales]),
            quats=torch.cat([self.quats, other.quats]),
            opacities=torch.cat([self.opacities, other.opacities]),
            sh_0=torch.cat([self.sh_0, other.sh_0]),
            sh_n=torch.cat([self.sh_n, other.sh_n]),
            times=torch.cat([self.times, other.times]),
            durations=torch.cat([self.durations, other.durations]),
            velocities=torch.cat([self.velocities, other.velocities]),
        )

    def __len__(self) -> int:
        return len(self.means)

    def save(self, path: str):
        torch.save(self, path)

    def save_dict(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, dict):
            return cls(**obj)
            
