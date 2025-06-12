# Local Laplacian Filter in PyTorch

## Overview

This project implements a **vectorized version** of the Local Laplacian Filter algorithm in **PyTorch**, optimized for GPU processing and batched image support. The original method is known for preserving edges while performing detail and tone manipulations using a multi-scale Laplacian pyramid.

---

## Features

- Fully **vectorized PyTorch implementation**
- GPU-compatible
- Supports **LDR and HDR** images
- Multi-scale edge-aware detail remapping
- Based on the **SIGGRAPH 2011** publication

---

## Visual Comparison

Below is a sample comparison between the **original image** and the result produced by the **Local Laplacian Filter**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/aa66e550-43ad-4820-8a42-36ef512d6fc9" width="45%" alt="Original">
  <img src="https://github.com/user-attachments/assets/f3eebb2d-64d1-475d-a418-d58756d43cd6" width="45%" alt="Filtered">
</p>

<p align="center"><i>Original vs. Filtered with Local Laplacian (detail enhancement)</i></p>

---

## Installation

```bash
git clone https://github.com/catenacciA/vectorlaplace.git
cd vectorlaplace
pip install torch numpy opencv-python tqdm
````

---

## Usage

```
script.py [-h] [--input INPUT] [--output OUTPUT] [--sigma_r SIGMA_R] [--alpha ALPHA] [--beta BETA] [--mode {auto,rgb,lum}] [--domain {lin,log}]
                 [--scale SCALE] [--levels LEVELS] [--device DEVICE] [--silent]

Laplacian Pyramid Edge-Aware Filter

options:
  -h, --help            show this help message and exit
  --input INPUT         Input path
  --output OUTPUT       Output path
  --sigma_r SIGMA_R     Detail threshold
  --alpha ALPHA         Detail preservation
  --beta BETA           Edge enhancement
  --mode {auto,rgb,lum}
  --domain {lin,log}
  --scale SCALE         Percent size
  --levels LEVELS       Pyr levels
  --device DEVICE       cpu or cuda
  --silent              No prints
```

---

## Reference

* Paris, S., Hasinoff, S.W., Kautz, J.
  [Local Laplacian Filters](https://people.csail.mit.edu/sparis/publi/2011/siggraph/#:~:text=ToG%20paper%20(more%20formal)%20(pdf%2C%2053MB)), SIGGRAPH 2011
* [Communications of the ACM, Vol. 58, No. 3](https://cacm.acm.org/research/local-laplacian-filters/)
