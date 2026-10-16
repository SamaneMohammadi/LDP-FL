import os
import glob
import argparse
import numpy as np

import config

EMOTION_TO_LABEL = {e: i for i, e in enumerate(config.EMOTIONS)}  # ANG/HAP/NEU/SAD -> 0..3


def extract_features(wav_paths):
    """Run OpenSMILE emobase on each wav, return (features, labels, speakers)."""
    import opensmile  # imported here so the rest of the repo doesn't need it

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feats, labels, speakers = [], [], []
    for path in wav_paths:
        name = os.path.basename(path).split("_")
        speaker, emotion = name[0], name[2]
        if emotion not in EMOTION_TO_LABEL:
            continue
        df = smile.process_file(path)          # 1 x 988
        feats.append(df.to_numpy().ravel().astype(np.float32))
        labels.append(EMOTION_TO_LABEL[emotion])
        speakers.append(speaker)
    return np.array(feats), np.array(labels, dtype=np.int64), np.array(speakers)


def speaker_partition(speakers, rng):
    """One client per speaker (the paper's setup): near-equal client sizes."""
    parts = []
    for sp in sorted(set(speakers)):
        idx = np.where(speakers == sp)[0]
        rng.shuffle(idx)
        parts.append(idx)
    return parts


def skewed_partition(n_samples, n_clients, beta, min_size, rng):
    """Quantity skew: client sizes ~ min_size + Dirichlet(beta) share of the rest
    (Li et al., 2022). Smaller beta -> more unequal sizes. Samples are assigned
    at random, so only the amount of data differs, not the label mix."""
    idx = rng.permutation(n_samples)
    props = rng.dirichlet(np.full(n_clients, beta))
    sizes = min_size + rng.multinomial(n_samples - n_clients * min_size, props)
    return np.split(idx, np.cumsum(sizes)[:-1])


def save_partitions(feats, labels, parts, out_dir, val_split=config.VAL_SPLIT):
    os.makedirs(out_dir, exist_ok=True)
    splits = []
    for idx in parts:
        n_val = max(1, int(len(idx) * val_split))
        splits.append((idx[n_val:], idx[:n_val]))

    # z-normalize with training-split statistics (raw emobase values span
    # several orders of magnitude and make SGD diverge)
    all_train = np.concatenate([tr for tr, _ in splits])
    mean = feats[all_train].mean(axis=0)
    std = feats[all_train].std(axis=0)
    std[std == 0] = 1.0
    feats = ((feats - mean) / std).astype(np.float32)

    for cid, (train_idx, val_idx) in enumerate(splits):
        np.save(os.path.join(out_dir, f"client_{cid}_x_train.npy"), feats[train_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_y_train.npy"), labels[train_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_x_val.npy"), feats[val_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_y_val.npy"), labels[val_idx])

    n_train = sorted(len(tr) for tr, _ in splits)
    print(f"wrote {len(splits)} client partitions to {out_dir}")
    print(f"train samples per client: min {n_train[0]}  median {n_train[len(n_train) // 2]}  "
          f"max {n_train[-1]}  total {sum(n_train)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="CREMA-D/AudioWAV")
    parser.add_argument("--out_dir", default="client_data")
    parser.add_argument("--features_cache", default="features_emobase.npz",
                        help="raw extracted features are cached here and reused")
    parser.add_argument("--partition", choices=["speaker", "skewed"], default="speaker")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Dirichlet concentration for --partition skewed")
    parser.add_argument("--min_size", type=int, default=10,
                        help="minimum samples per client for --partition skewed")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if os.path.exists(args.features_cache):
        d = np.load(args.features_cache)
        feats, labels, speakers = d["feats"], d["labels"], d["speakers"]
        print(f"loaded cached features {feats.shape} from {args.features_cache}")
    else:
        wavs = sorted(glob.glob(os.path.join(args.data_path, "*.wav")))
        print(f"found {len(wavs)} wav files")
        feats, labels, speakers = extract_features(wavs)
        np.savez(args.features_cache, feats=feats, labels=labels, speakers=speakers)
    print(f"features: {feats.shape} over {len(set(speakers))} speakers")

    rng = np.random.default_rng(args.seed)
    if args.partition == "speaker":
        parts = speaker_partition(speakers, rng)
    else:
        parts = skewed_partition(len(feats), config.NUM_CLIENTS, args.beta, args.min_size, rng)
    save_partitions(feats, labels, parts, args.out_dir)
