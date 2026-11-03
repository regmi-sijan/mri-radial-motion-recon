# Motion-resolved radial MRI simulation

Milestone 1 establishes a static reconstruction baseline for a free-breathing MRI simulation study. The synthetic coronal slice contains a liver, kidney, spine, lung and small liver lesion. Inspiration moves the liver and lesion inferiorly. The kidney moves 40% as far. The breathing model varies the amplitude and duration between cycles using a fixed seed.

## Milestone 1 results

| Check | Measured | Target |
| --- | ---: | ---: |
| NUFFT adjoint relative error | 8.95e-16 | < 1e-4 |
| Static gridding closure NRMSE | 3.21% | < 5% |

![Abdominal phantom at two breathing phases and its simulated displacement](results/figures/F1_phantom.png)

![First 30 golden-angle radial spokes](results/figures/F2_trajectory.png)

## Run M1

The following commands run from the repository root. The pinned versions were checked with Python 3.14.2.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.milestone1
```

The figures appear under `results/figures` as PNG and PDF. In F1, compare the liver and bright lesion between breathing extremes and check that the spine stays fixed. In F2, every spoke passes through the origin and the angles spread across the disk.

## Static reconstruction

K-space stores spatial frequencies. Each coil measures the Fourier transform of the phantom multiplied by its complex sensitivity. The synthetic sensitivities have unit summed squared magnitude at every pixel. For radial sampling, each spoke includes the zero-frequency sample. Stepping by 111.246° spreads successive spokes around the disk.

The forward operator pads each coil image, computes an orthonormal FFT, and interpolates the result at the radial coordinates. Its adjoint scatters the same interpolation weights onto the Fourier grid, applies the inverse FFT, and crops back to the image. The adjoint test checks the complex dot-product identity. The gridding baseline applies a radial density weight proportional to radius, then combines coil images with conjugate sensitivities.

On the local CPU the adjoint test gave a relative error of 8.95e-16. Static closure with 402 spokes and zero added noise gave 3.21% NRMSE after fitting one global intensity scale. This measures closure with a matched simulator and reconstruction operator. The phantom edges include 0.85-pixel Gaussian smoothing. The operator uses SciPy with bilinear interpolation on a fourfold oversampled FFT grid. TOML and standard-library unittest keep the first checkpoint runnable with the available environment.

The R3 portion of static closure belongs to M3. Motion-dependent k-space, noise calibration and respiratory binning belong to M2. All data here are synthetic.
