# Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics

Much of this code is an adapted version of the LagrangeBench code available at https://github.com/tumaer/lagrangebench. Our core functions are the following two:
- `relax_wrapper` in `neural_sph/rollout.py` - core relaxation algorithm
- `case_setup_redist` in `neural_sph/utils.py` - setup needed by the relaxation routine including neighbor list preallocation, box size for periodic boundary conditions, etc. 

## Install

```bash
# set up environment and install dependencies
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # install dependencies
pip install -e .  # install neuralsph as a package

# on a cuda12 machine run this line in addition:
# pip install --upgrade jax[cuda12_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Pre-trained models
Run `python download_checkpoints.py` to download all pre-trained model weights from LagrangeBench.

### Datasets
Run `bash download_data.sh all datasets/` to download all datasets from LagrangeBench.

### Run the code
We show how to obtain the numbers we get on a rollout with our SPH relaxation on the example of a 2D LDC checkpoint stored in `CKP_DIR=ckp/pretrained/gns_ldc2d/best`:

```bash
# baseline without relaxation
python main.py eval.test=True load_ckp=$CKP_DIR eval.infer.n_trajs=-1 \
    eval.infer.metrics_stride=10 eval.n_rollout_steps=400 \
    r.variant_p=None

# rollout including 5-step relaxation with $\alpha=0.03$
python main.py eval.test=True load_ckp=$CKP_DIR eval.infer.n_trajs=-1 \
    eval.infer.metrics_stride=10 eval.n_rollout_steps=400 \
    r.loops=5 r.acc=0.03
```

## RPF Force Smoothing

To implement the external force smoothing on the reverse Poiseuille datasets, we replaced the force function in the `force.py` file contained within the LagrangeBench datasets by the following two functions.

2D:
```python
def force_fn(r):
    """Smoothed version of the 2D RPF force function using the error function"""
    sigma = 0.025 
    erf_mitte = lax.erf((r[1] - 1) / (jnp.sqrt(2) * sigma))
    erf_left = lax.erf(r[1] / (jnp.sqrt(2) *sigma)) 
    erf_right = lax.erf((r[1] - 2) / (jnp.sqrt(2) * sigma)) 
    res = erf_left + erf_right - erf_mitte
    return jnp.array([res, 0.0])
```

3D:
```python
def force_fn(r):
    """Smoothed version of the 3D RPF force function using the error function"""
    sigma = 0.05
    erf_mitte = lax.erf((r[1] - 1) / (jnp.sqrt(2) * sigma))
    erf_left = lax.erf(r[1] / (jnp.sqrt(2) *sigma)) 
    erf_right = lax.erf((r[1] - 2) / (jnp.sqrt(2) * sigma)) 
    res = erf_left + erf_right - erf_mitte
    return jnp.array([res, 0.0, 0.0])   
```

For simplicity, we added the external force functions for RPF and dam break directly to the `case.py > case_builder` and one can configure them with the CLI arguments `r.is_subtract_ext_force` and `r.is_smooth_force`, see `defaults.py` for more details.

## Citation
Cite the [ICML 2024 paper](https://arxiv.org/abs/2402.06275) as
```bibtex
@article{toshev2024neural,
  title={Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics},
  author={Toshev, Artur P and Erbesdobler, Jonas A and Adams, Nikolaus A and Brandstetter, Johannes},
  journal={arXiv preprint arXiv:2402.06275},
  year={2024}
}
```