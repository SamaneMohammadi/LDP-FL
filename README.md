# LDP-FL with CSS

Implementation of **Balancing Privacy and Accuracy in Federated Learning for
Speech Emotion Recognition** (FedCSIS 2023).

📄 Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10306049)

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
  runs FedSGD `g̃ = (1/K) Σ g̃_i`, `W ← W − η g̃`. Cumulative privacy `ε` per
  client is tracked with the **Moments Accountant**.
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
```

The model inversion attack is exposed as `invert(model, target_label)` in
`src/attack.py` for evaluating a trained model's privacy robustness.

## Citation

```bibtex
@inproceedings{mohammadi2023balancing,
  title={Balancing Privacy and Accuracy in Federated Learning for Speech Emotion Recognition},
  author={Mohammadi, Samaneh and Mohammadi, Mohammadreza and Sinaei, Sima and Balador, Ali and Nowroozi, Ehsan and Flammini, Francesco and Conti, Mauro},
  booktitle={2023 18th Conference on Computer Science and Intelligence Systems (FedCSIS)},
  pages={191--199},
  year={2023},
  organization={IEEE}
}
```

## License

MIT.
