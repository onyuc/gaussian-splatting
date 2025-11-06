import torch
from _gaussians import DynamicGaussians
from pathlib import Path
import gsplat

def entropy_indices(x: torch.Tensor,
                                    entropy_threshold: float = 0.75,
                                    eps: float = 1e-12):
    """
    Returns:
    idx_entropy: indices where normalized entropy >= entropy_threshold
    """

    N, T = x.shape
    logT = torch.log(torch.tensor(float(T), dtype=x.dtype, device=x.device))

    # --- (1) Entropy uniformity ---
    row_sum = x.sum(dim=1, keepdim=True) + eps           # [N,1]
    p = x / row_sum                                       # [N,T], sums to 1 per row (normalized)
    H = -(p * (p.add_(eps).log())).sum(dim=1)            # entropy per row
    u = H / logT                                          # normalized entropy in [0,1]
    idx_entropy = u >= entropy_threshold

    return idx_entropy

def evenly_distributed_bool(gs: DynamicGaussians,
                               entropy_threshold: float = 0.75,
                               eps: float = 1e-12):
    """
    Returns:
    idx_entropy: bool where normalized entropy >= entropy_threshold
    """
    fps = 30
    
    opacity_list = [gs.temporal_opacity(idx / fps).squeeze(-1) for idx in range(300)]
    opacities = torch.stack(opacity_list, dim=1)
    return entropy_indices(opacities, entropy_threshold, eps)

if __name__ == "__main__":
    frame_idx = 2

    fps = 30
    t = frame_idx / fps
    gs = DynamicGaussians.load("gaussians_dict.pt")
    opacity_bool = gs.temporal_opacity(t).squeeze(-1) > 0.005

    static_bool = evenly_distributed_bool(gs)
    # static_bool = evenly_distributed_bool(gs, entropy_threshold=0.85)
    # static_bool = evenly_distributed_bool(gs, entropy_threshold=0.95)

    means = gs.temporal_means(t)[static_bool]
    scales = gs.scales[static_bool]
    quats = gs.quats[static_bool]
    opacities = gs.temporal_opacity_logit(t).squeeze(-1)[static_bool]
    sh_0 = gs.sh_0[static_bool]
    sh_n = gs.sh_n[static_bool]

    file_path = Path("./statics")
    file_path.mkdir(parents=True, exist_ok=True)

    # opacity
    gsplat.export_splats(
        means=gs.temporal_means(t)[opacity_bool],
        scales=gs.scales[opacity_bool],
        quats=gs.quats[opacity_bool],
        opacities=gs.temporal_opacity_logit(t).squeeze(-1)[opacity_bool],
        sh0=gs.sh_0[opacity_bool],
        shN=gs.sh_n[opacity_bool],
        format="ply",
        save_to=str(file_path / f"opacity_{frame_idx:03d}.ply"),
    )

    # # static
    # gsplat.export_splats(
    #     means=means,
    #     scales=scales,
    #     quats=quats,
    #     opacities=opacities,
    #     sh0=sh_0,
    #     shN=sh_n,
    #     format="ply",
    #     save_to=str(file_path / f"static_{frame_idx:03d}.ply"),
    # )

    # d_means = gs.temporal_means(t)[~static_idx]
    # d_scales = gs.scales[~static_idx]
    # d_quats = gs.quats[~static_idx]
    # d_opacities = gs.temporal_opacity_logit(t).squeeze(-1)[~static_idx]
    # d_sh_0 = gs.sh_0[~static_idx]
    # d_sh_n = gs.sh_n[~static_idx]

    # gsplat.export_splats(
    #     means=d_means,
    #     scales=d_scales,
    #     quats=d_quats,
    #     opacities=d_opacities,
    #     sh0=d_sh_0,
    #     shN=d_sh_n,
    #     format="ply",
    #     save_to=str(file_path / f"dynamic_{frame_idx:03d}.ply"),
    # )