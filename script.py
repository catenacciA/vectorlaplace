import math
import time
import cv2
from typing import List, Optional, Tuple

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Image I/O (OpenCV ONLY used here)
# ──────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {path}")

    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img


def save_image(path: str, img: np.ndarray, hdr: bool = False, silent: bool = False):

    if hdr:
        img = img.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=1e3, neginf=0.0)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        ldr = np.clip(img, 0.0, 1.0)
        ldr_uint8 = (ldr * 255.0).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(ldr_uint8, cv2.COLOR_RGB2BGR))

    if not silent:
        print(f"Image saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────


def reflect_pad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Reflect pad even when padding exceeds PyTorch's internal constraints.
    Workaround for https://github.com/pytorch/vision/issues/8622
    """
    l, r, t, b = pad
    paddings = torch.tensor([l, r, t, b], dtype=torch.int32)
    assert torch.all(paddings >= 0), "Negative padding not supported"
    while torch.any(paddings > 0):
        H, W = x.shape[-2:]
        limits = torch.tensor([W - 1, W - 1, H - 1, H - 1])
        chunk = torch.minimum(paddings, limits)
        x = F.pad(x, tuple(chunk.tolist()), mode="reflect")
        paddings -= chunk
    return x


def child_window(win):
    y0, y1, x0, x1 = win
    row_off = y0 & 1
    col_off = x0 & 1

    child = (
        (y0 + row_off) // 2,
        (y1 - row_off) // 2,
        (x0 + col_off) // 2,
        (x1 - col_off) // 2,
    )
    return child, row_off, col_off


# ──────────────────────────────────────────────────────────────────────────────
#  Gaussian-blur helpers
# ──────────────────────────────────────────────────────────────────────────────


def binomial_kernel(
    channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    a = torch.tensor([0.05, 0.25, 0.40, 0.25, 0.05], device=device, dtype=dtype)
    k2d = a[:, None] * a[None, :]
    return k2d.unsqueeze(0).expand(channels, 1, 5, 5).contiguous()


def _gauss_filter_norm(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = (2, 2, 2, 2)
    x_pad = reflect_pad(x, pad)
    C = x.shape[1]
    ones_c = torch.ones_like(x)
    ones_pad = reflect_pad(ones_c, pad)
    numer = F.conv2d(x_pad, kernel, groups=C)
    denom = F.conv2d(ones_pad, kernel, groups=C)
    denom = torch.clamp(denom, min=1e-8)
    return numer / denom


def _gauss_filter_masked(x, mask, kernel):
    pad = (2, 2, 2, 2)
    x_pad = reflect_pad(x * mask, pad)
    C = x.shape[1]
    mask_c = mask.expand(-1, C, -1, -1)
    m_pad = reflect_pad(mask_c, pad)
    numer = F.conv2d(x_pad, kernel, groups=C)
    denom = F.conv2d(m_pad, kernel, groups=C)
    denom = torch.clamp(denom, min=1e-8)
    return numer / denom


# ──────────────────────────────────────────────────────────────────────────────
#  Down- / up-sampling
# ──────────────────────────────────────────────────────────────────────────────


def _downsample(x, kernel, win):
    y0, y1, x0, x1 = win
    row_off = y0 & 1
    col_off = x0 & 1
    x_f = _gauss_filter_norm(x, kernel)
    res = x_f[..., y0 + row_off : y1 + 1 : 2, x0 + col_off : x1 + 1 : 2]

    child_win, child_row_off, child_col_off = child_window(win)
    return res, child_win, child_row_off, child_col_off


def _upsample(x, parent_win, kernel):
    y0, y1, x0, x1 = parent_win
    Ht = y1 - y0 + 1
    Wt = x1 - x0 + 1

    B, C = x.shape[:2]
    up = torch.zeros(B, C, Ht, Wt, device=x.device, dtype=x.dtype)
    mask = torch.zeros_like(up[:, :1])

    row_even = y0 & 1
    col_even = x0 & 1
    up[..., row_even::2, col_even::2] = x
    mask[..., row_even::2, col_even::2] = 1.0

    return _gauss_filter_masked(up, mask, kernel)


# ──────────────────────────────────────────────────────────────────────────────
#  Detail & edge remapping functions
# ──────────────────────────────────────────────────────────────────────────────


def smooth_step(xmin: float, xmax: float, x: torch.Tensor) -> torch.Tensor:
    y = torch.clamp((x - xmin) / (xmax - xmin), 0.0, 1.0)
    return y**2 * (y - 2.0) ** 2


def fd(d: torch.Tensor, alpha: float, sigma_r: float) -> torch.Tensor:
    out = d**alpha
    if alpha < 1.0:
        noise = 0.01
        tau = smooth_step(noise, 2 * noise, d * sigma_r)
        out = tau * out + (1.0 - tau) * d
    return out


def fe(a: torch.Tensor, beta: float) -> torch.Tensor:
    return beta * a


def r_gray(patches, g0, sigma_r, alpha, beta):
    diff = patches - g0
    dnrm = diff.abs()
    sign = diff.sign()
    rd = g0 + sign * sigma_r * fd(dnrm / sigma_r, alpha, sigma_r)
    re = g0 + sign * (fe(dnrm - sigma_r, beta) + sigma_r)
    return torch.where(dnrm > sigma_r, re, rd)


def r_color(patches, g0, sigma_r, alpha, beta):
    diff = patches - g0
    dnrm = torch.sqrt((diff**2).sum(dim=1, keepdim=True))
    eps = torch.finfo(patches.dtype).eps
    unit = diff / (dnrm + eps)
    rd = g0 + unit * sigma_r * fd(dnrm / sigma_r, alpha, sigma_r)
    re = g0 + unit * (sigma_r + fe(dnrm - sigma_r, beta))
    return torch.where(dnrm > sigma_r, re, rd)


# ──────────────────────────────────────────────────────────────────────────────
#  Pyramid helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_num_levels(image: torch.Tensor) -> int:
    rows, cols = image.shape[-2], image.shape[-1]
    levels = 1
    while min(rows, cols) > 1:
        levels += 1
        rows, cols = (rows + 1) // 2, (cols + 1) // 2
    return levels


def build_gauss_pyr(img: torch.Tensor, levels: int, kernel: torch.Tensor):
    pyr: List[torch.Tensor] = [img]
    subs: List[Tuple[Tuple[int, int, int, int], int, int]] = []

    H, W = img.shape[-2:]
    init_win = (0, H - 1, 0, W - 1)
    subs.append((init_win, 0, 0))

    for i in range(1, levels):
        prev, (win, _, _) = pyr[-1], subs[-1]
        g, child_win, row_off, col_off = _downsample(prev, kernel, win)
        pyr.append(g)
        subs.append((child_win, row_off, col_off))
    return pyr, subs


# ──────────────────────────────────────────────────────────────────────────────
#  Core filtering  (vectorised)
# ──────────────────────────────────────────────────────────────────────────────


def lapfilter_core(
    image: torch.Tensor,
    num_levels: int,
    kernel: torch.Tensor,
    sigma_r: float,
    alpha: float,
    beta: float,
    is_color: bool,
    show_progress: bool,
):
    gauss_pyr, windows = build_gauss_pyr(image, num_levels, kernel)
    laplacian = [torch.zeros_like(l) for l in gauss_pyr]

    out_dir = "debug_levels"
    os.makedirs(out_dir, exist_ok=True)

    pbar = (
        tqdm(
            total=num_levels - 1,
            desc="Filtering",
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}",
            colour="green",
        )
        if show_progress
        else None
    )

    for lvl in range(num_levels - 1):
        base = gauss_pyr[lvl]
        (win, row_off, col_off) = windows[lvl]
        y0_win, _, x0_win, _ = win
        B, C, H, W = base.shape
        pr_cap = (min(H, W) - 1) // 2
        pr = min(3 * (1 << lvl), pr_cap)
        ps = 2 * pr + 1

        if pr == 0:
            laplacian[lvl].zero_()
            if pbar is not None:
                pbar.update(1)
            continue

        pad = (pr, pr, pr, pr)

        padded = reflect_pad(base, pad)
        patches = F.unfold(padded, kernel_size=ps).view(B, C, ps, ps, -1)

        mask_pad = reflect_pad(torch.ones_like(base[:, :1]), pad)
        mask_patches = F.unfold(mask_pad, kernel_size=ps).view(B, 1, ps, ps, -1)

        centre_orig = base.reshape(B, C, -1).unsqueeze(-2).unsqueeze(-2)

        remapped = (r_color if is_color else r_gray)(
            patches, centre_orig, sigma_r, alpha, beta
        )

        seq = remapped.permute(0, 4, 1, 2, 3).reshape(-1, C, ps, ps)
        mseq = mask_patches.permute(0, 4, 1, 2, 3).reshape(-1, 1, ps, ps)

        g0 = _gauss_filter_masked(seq, mseq, kernel)

        par_row = row_off
        par_col = col_off

        ctr_row = pr if par_row == 0 else pr - 1
        ctr_col = pr if par_col == 0 else pr - 1

        g1 = g0[..., par_row::2, par_col::2]

        parent_win = (y0_win, y0_win + ps - 1, x0_win, x0_win + ps - 1)
        g1_up = _upsample(g1, parent_win, kernel)

        centre_low = g1_up[..., ctr_row, ctr_col]
        centre_low = (
            centre_low.view(B, H * W, C).permute(0, 2, 1).view(B, C, 1, 1, H * W)
        )

        centre_remap = remapped[:, :, pr, pr, :].view(B, C, 1, 1, H * W)
        laplacian[lvl] = (centre_remap - centre_low).view(B, C, H, W)

        lvl_img = laplacian[lvl]
        fname = os.path.join(out_dir, f"port_l{lvl+1}.png")
        arr = lvl_img.squeeze(0)
        if arr.shape[0] == 1:
            img_np = arr.squeeze(0).clamp(0, 1).cpu().numpy()
        else:
            img_np = arr.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img_np = (img_np * 255.0).astype(np.uint8)
        cv2.imwrite(fname, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(level=f"{H}×{W}", patch=ps)

    if pbar is not None:
        pbar.close()

    laplacian[num_levels - 1] = gauss_pyr[num_levels - 1]
    arr = laplacian[num_levels - 1].squeeze(0)

    if arr.shape[0] == 1:
        img_np = arr.squeeze(0).clamp(0, 1).cpu().numpy()
    else:
        img_np = arr.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    img_np = (img_np * 255.0).astype(np.uint8)
    fname = os.path.join(out_dir, f"port_l{num_levels}.png")
    cv2.imwrite(fname, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    return laplacian, windows


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────


def lapfilter(
    img_np: np.ndarray,
    *,
    sigma_r: float = 0.4,
    alpha: float = 0.25,
    beta: float = 1.0,
    color_mode: str = "auto",
    domain: str = "lin",
    scale: int = 100,
    num_levels: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:

    t0 = time.time()

    orig_dtype = img_np.dtype
    orig_shape = img_np.shape
    alpha_np = None

    if img_np.ndim == 2:
        img_np = img_np[:, :, None]

    if img_np.ndim == 3 and img_np.shape[2] == 4:
        alpha_np = img_np[:, :, 3].copy()
        img_np = img_np[:, :, :3]

    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)

    img = img_np.astype(np.float32)
    if orig_dtype == np.uint8:
        img = img / 255.0

    H, W, C = img.shape
    if verbose:
        print(f"\nProcessing image: {H}×{W}, channels={C}")

    if color_mode == "auto":
        color_mode = "rgb" if C >= 3 else "lum"
    if verbose:
        print(f"Color mode: {color_mode}")

    Iratio_np = None
    if color_mode == "lum":
        IY_np = (20 * img[:, :, 0] + 40 * img[:, :, 1] + 1 * img[:, :, 2]) / 61.0
        eps_np = np.finfo(np.float32).eps
        Iratio_np = img / np.expand_dims(IY_np + eps_np, axis=2)
        img = IY_np[:, :, None]
        if verbose:
            print("Converted to luminance IY for 'lum' mode")

    tensor = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    if scale != 100:
        Ht, Wt = tensor.shape[-2], tensor.shape[-1]
        new_H = int(round(Ht * scale / 100))
        new_W = int(round(Wt * scale / 100))
        if verbose:
            print(f"Resizing to {new_H}×{new_W}")
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode="area")

    eps_torch = torch.finfo(tensor.dtype).eps

    sigma_r_mapped = sigma_r
    if domain == "log":
        sigma_r_mapped = math.log(max(sigma_r, eps_torch))
        if verbose:
            print(f"domain='log': remapped σ_r ← log(σ_r) = {sigma_r_mapped:.4f}")

    if domain == "lin":
        to_domain = lambda I: I
        from_domain = lambda R: R
    elif domain == "log":
        to_domain = lambda I: torch.log(torch.clamp(I, min=eps_torch))
        from_domain = lambda R: torch.exp(R)
        if verbose:
            print("Using log-domain transforms")
    else:
        raise ValueError("Invalid domain: choose 'lin' or 'log'")

    skip_filter = alpha == 1.0 and beta == 1.0

    if not skip_filter:
        tensor = to_domain(tensor)
        is_rgb = color_mode == "rgb" and tensor.shape[1] >= 3

        levels = num_levels or _get_num_levels(tensor)
        if verbose:
            print(
                f"Pyramid levels: {levels}, σ_r={sigma_r_mapped:.4f}, α={alpha}, β={beta}"
            )

        kernel = binomial_kernel(
            channels=tensor.shape[1], device=tensor.device, dtype=tensor.dtype
        )

        if verbose:
            print("\nRunning lapfilter_core…")
        lap_pyr, subs = lapfilter_core(
            tensor,
            levels,
            kernel,
            sigma_r_mapped,
            alpha,
            beta,
            is_rgb,
            show_progress=verbose,
        )

        if verbose:
            print("\nReconstructing from pyramid…")
        out_tensor = lap_pyr[-1]
        for lev in range(levels - 1, 0, -1):
            parent_win, *_ = subs[lev - 1]
            out_tensor = _upsample(out_tensor, parent_win, kernel) + lap_pyr[lev - 1]

        out_tensor = from_domain(out_tensor)

    else:
        if domain == "log":
            out_tensor = from_domain(to_domain(tensor))
        else:
            out_tensor = tensor
        if verbose:
            print("α=1 & β=1 → skipping lapfilter_core (identity)")

    if verbose:
        elapsed = time.time() - t0
        print(f"\nFiltering done in {elapsed:.2f}s")

    if domain == "log" and beta <= 1.0:
        if color_mode == "lum":
            Y = out_tensor[:, 0, :, :]
        else:
            wts = (
                torch.tensor(
                    [20.0, 40.0, 1.0], device=out_tensor.device, dtype=out_tensor.dtype
                )
                / 61.0
            )
            Y = (out_tensor[:, 0:3, :, :] * wts.view(1, 3, 1, 1)).sum(dim=1)
        Y_cpu = Y.detach().cpu().numpy().flatten()
        prc_low = 0.5
        prc_high = 100.0 - 0.5
        Rmin_clip = np.percentile(Y_cpu, prc_low)
        Rmax_clip = np.percentile(Y_cpu, prc_high)
        Rmin_clip = max(Rmin_clip, np.finfo(np.float32).eps)
        Rmax_clip = max(Rmax_clip, np.finfo(np.float32).eps)
        DR_clip = Rmax_clip / Rmin_clip
        DR_desired = 100.0
        exponent = math.log(DR_desired) / math.log(DR_clip)
        if verbose:
            print(
                f"Log-domain tone mapping: Rmin={Rmin_clip:.6f}, Rmax={Rmax_clip:.6f}, exponent={exponent:.4f}"
            )

        out_tensor = torch.clamp(out_tensor, min=0.0) / float(Rmax_clip)
        out_tensor = out_tensor.pow(exponent)

    if color_mode == "lum":
        out_tensor = out_tensor.repeat(1, 3, 1, 1)
        if scale != 100:
            Iratio_t = (
                torch.from_numpy(Iratio_np)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            Iratio_t = F.interpolate(
                Iratio_t,
                size=(out_tensor.shape[-2], out_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            Iratio_resized = Iratio_t.squeeze(0)
        else:
            Iratio_resized = (
                torch.from_numpy(Iratio_np)
                .permute(2, 0, 1)
                .to(device=device, dtype=torch.float32)
            )
        out_tensor = out_tensor * Iratio_resized.unsqueeze(0)
        if verbose:
            print("Reapplied color ratios for 'lum' mode")

    out_tensor = torch.clamp(out_tensor, min=0.0)

    is_uint8_input = orig_dtype == np.uint8
    wants_tonemapped_ldr = domain == "log" and beta <= 1.0

    if is_uint8_input or wants_tonemapped_ldr:
        out_tensor = torch.clamp(out_tensor, max=1.0)

    if domain == "log" and beta <= 1.0:
        gamma_val = 2.2
        out_tensor = out_tensor.pow(1.0 / gamma_val)
        if verbose:
            print("Applied gamma correction (1/2.2)")

    out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if orig_shape[-1] == 1 or orig_shape == img_np.shape[:2]:
        out_np = out_np[:, :, 0]

    if alpha_np is not None:
        if scale != 100:
            alpha_t = torch.from_numpy(
                alpha_np.astype(np.float32)[None, None, :, :]
            ).to(device=device)
            alpha_t = F.interpolate(
                alpha_t, size=(out_np.shape[0], out_np.shape[1]), mode="nearest"
            )
            alpha_resized = alpha_t.squeeze().cpu().numpy().astype(alpha_np.dtype)
        else:
            alpha_resized = alpha_np
        if out_np.ndim == 2:
            out_np = out_np[:, :, None]
        out_np = np.dstack([out_np, alpha_resized])

    return out_np.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Laplacian Pyramid Edge-Aware Filter")
    parser.add_argument(
        "--input", default="input_images/input_png/flower.png", help="Input path"
    )
    parser.add_argument("--output", default="port_out.png", help="Output path")
    parser.add_argument("--sigma_r", type=float, default=0.4, help="Detail threshold")
    parser.add_argument("--alpha", type=float, default=0.25, help="Detail preservation")
    parser.add_argument("--beta", type=float, default=1.0, help="Edge enhancement")
    parser.add_argument("--mode", choices=["auto", "rgb", "lum"], default="rgb")
    parser.add_argument("--domain", choices=["lin", "log"], default="lin")
    parser.add_argument("--scale", type=int, default=100, help="Percent size")
    parser.add_argument("--levels", type=int, default=None, help="Pyr levels")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--silent", action="store_true", help="No prints")
    args = parser.parse_args()

    if not args.silent:
        print("\n" + "=" * 40)
        print("Laplacian Pyramid Edge-Aware Filter")
        print("=" * 40)

    try:
        img = load_image(args.input)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    torch.set_num_threads(os.cpu_count())

    is_hdr_output = args.output.lower().endswith(".hdr")

    out = lapfilter(
        img,
        sigma_r=args.sigma_r,
        alpha=args.alpha,
        beta=args.beta,
        color_mode=args.mode,
        domain=args.domain,
        scale=args.scale,
        num_levels=args.levels,
        device=args.device,
        verbose=not args.silent,
    )

    try:
        save_image(args.output, out, hdr=is_hdr_output, silent=args.silent)
    except Exception as e:
        print(f"Error saving: {e}")
        return

    if not args.silent:
        print(f"Saved to {args.output}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
