# Balancing Privacy and Accuracy in Federated Learning for Speech Emotion Recognition

📄 Paper: [FedCSIS 2023](https://ieeexplore.ieee.org/document/10306049)

Local Differential Privacy (LDP) protects clients' speech data in federated SER
by adding noise to model parameters before they leave the device, but that noise
degrades accuracy. This work proposes **LDP-FL with CSS**, combining LDP with a
novel Client Selection Strategy (CSS) that makes each round's updates more
representative — and therefore more robust to noise — by favouring clients with
larger local datasets while keeping diversity through random selection. The
approach is evaluated against model inversion attacks, which attempt to
reconstruct a client's speech features from the model's output labels: LDP-FL
with CSS keeps accuracy within ~4% of the non-private SER model while proving far
more resilient to inversion than the non-LDP baseline.

## Methods

- **LDP (DP-SGD)** — each client clips per-sample gradients to L2 norm `C`,
  averages them, and adds Gaussian noise `N(0, σ²C²I)` before sharing; the server
  runs FedSGD. Cumulative privacy `ε` per client is tracked with the **Moments Accountant**.
- **CSS (Client Selection Strategy)** — each round picks `K` clients: the top
  `K/2` by local sample size (more data → more representative, noise-robust
  updates) plus `K/2` chosen at random (diversity, no overlap). The `random`
  baseline is included for comparison.
- **Model inversion attack** — an adversary that knows a target emotion label and
  the client model reconstructs the speech features by gradient descent on the
  input to minimize `c(x) = 1 − f_label(x)`, used to measure privacy robustness.
- **Model** — an MLP over OpenSMILE emobase features for four emotions
  (neutral, sad, happy, angry), trained with FedSGD.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src

# 1) prepare per-client SER data (CREMA-D -> emobase features)
python prepare_data.py --data_path ./CREMA-D/AudioWAV --out_dir ./client_data

# 2) train LDP-FL with CSS (vs the random-selection baseline)
python main.py --selection css    --sigma 1.0 --clip 2.0 --rounds 50
python main.py --selection random --sigma 1.0 --clip 2.0 --rounds 50

# quantity-skewed clients (Dirichlet sizes, beta=0.5 -> 8..272 samples/client),
# to see the effect of CSS on accuracy and per-client privacy
python prepare_data.py --partition skewed --beta 0.5 --out_dir ./client_data_skewed
python main.py --selection css --sigma 1.0 --data_dir client_data_skewed --out_dir results/skewed
python main.py --selection random --sigma 1.0 --data_dir client_data_skewed --out_dir results/skewed
python plot_privacy.py --results_dir results/skewed --tag sigma1.0_C2.0_delta1e-05_K50

# aggregate a multi-seed sweep laid out as results/sweep/<partition>_s<seed>/
# (mean +- std table in summary.md and sweep.png)
python summarize_sweep.py --root results/sweep

# non-LDP FL baseline (no clipping, no noise)
python main.py --no_dp --selection css --rounds 50

# 3) model inversion attack (Table I): K=7, C in {1,2,4}, sigma in {1,3,5,7,10}
python run_attack.py
```

**Privacy reporting.** Each client keeps its own Moments Accountant. A client's
epsilon depends on its data size (local steps Poisson-sample lots at rate
`q = B/n`, `n/B` steps per round) and on how many rounds it was selected for.
CSS selects the largest clients every round, so they spend the most budget. The
reported `(eps, delta)`-LDP guarantee is the **worst case over clients**;
`main.py` also prints epsilon for CSS-selected vs randomly selected clients and
writes `results/eps_clients_*.csv` (per client) and `results/eps_rounds_*.csv`
(per-round trajectory for every client).

The batch rule decides how size affects epsilon. By default every client uses
lots of `B` samples, so small clients (q = B/n close to 1) get no amplification and
spend the most per round. With `--lot_frac f` every client uses q = f, so
epsilon depends only on how often a client is selected, and CSS's always-selected
clients carry the worst-case budget.

`prepare_data.py` z-normalizes the emobase features with training-split
statistics. The model inversion attack is exposed as `invert(model, target_label)`
in `src/attack.py`.

## Citation

```bibtex
@inproceedings{mohammadi2023balancing,
  title={Balancing privacy and accuracy in federated learning for speech emotion recognition},
  author={Mohammadi, Samaneh and Mohammadi, Mohammadreza and Sinaei, Sima and Balador, Ali and Nowroozi, Ehsan and Flammini, Francesco and Conti, Mauro},
  booktitle={2023 18th Conference on Computer Science and Intelligence Systems (FedCSIS)},
  pages={191--199},
  year={2023},
  organization={IEEE}
}

```

## License

MIT.
