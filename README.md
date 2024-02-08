# Neural SPH: Improved Neural Modeling of Langrangian Fluid Dynamics

We note that much of the code we provide is an adapted version of the code available on https://github.com/tumaer/lagrangebench. Our core functions are the following two:
- `relax_wrapper` in `neural_sph/rollout.py` - core relaxation algorithm
- `case_setup_redist` in `neural_sph/utils.py` - setup needed by the relaxation routine including neighbor list preallocation, box size for periodic boundary conditions, etc. 

## Install

```bash
# set up environment and install dependencies
python3.10 -m venv venv
source venv/bin/activate
pip install lagrangebench --extra-index-url=https://download.pytorch.org/whl/cpu

# on a cuda12 machine run this line in addition:
# pip install --upgrade jax[cuda12_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Run the code

We note that to run the code, one would need to retrain the used models.
To train a model with LagrangeBench, please look at https://github.com/tumaer/lagrangebench.

We show how to obtain the numbers we get on a rollout with our SPH relaxation on the example of a 2D LDC checkpoint stored in `CKP_DIR`:


```bash
# baseline without relaxation
python main.py --mode=infer  --test --model_dir=$CKP_DIR \
    --eval_n_trajs_infer=-1 --metrics_stride_infer=10 --n_rollout_steps=400 \
    -rvp=None
 
# rollout including 5-step relaxation with $\alpha=0.03$
python main.py --mode=infer  --test --model_dir=$CKP_DIR \
    --eval_n_trajs_infer=-1 --metrics_stride_infer=10 --n_rollout_steps=400 \
    -rl=5 -ra=0.03
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