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


def save_per_speaker(feats, labels, speakers, out_dir, val_split=config.VAL_SPLIT, seed=config.SEED):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for cid, sp in enumerate(sorted(set(speakers))):
        idx = np.where(speakers == sp)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        np.save(os.path.join(out_dir, f"client_{cid}_x_train.npy"), feats[train_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_y_train.npy"), labels[train_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_x_val.npy"), feats[val_idx])
        np.save(os.path.join(out_dir, f"client_{cid}_y_val.npy"), labels[val_idx])
    print(f"wrote {len(set(speakers))} client partitions to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="CREMA-D/AudioWAV")
    parser.add_argument("--out_dir", default="client_data")
    args = parser.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.data_path, "*.wav")))
    print(f"found {len(wavs)} wav files")
    feats, labels, speakers = extract_features(wavs)
    print(f"extracted features: {feats.shape} over {len(set(speakers))} speakers")
    save_per_speaker(feats, labels, speakers, args.out_dir)
