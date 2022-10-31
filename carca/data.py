import numpy as np
import pickle
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset


def load_ctx(dataset_name) -> Dict[Tuple[int, int], np.ndarray]:
    with open(f"../data/{dataset_name}_ctx.dat", "rb") as rf:
        ctx = pickle.load(rf)

    # Cast context values from list to numpy array
    for k in ctx.keys():
        ctx[k] = np.array(ctx[k], dtype=np.float32)

    return ctx


def load_attrs(dataset_name) -> np.ndarray:
    with open(f"../data/{dataset_name}_attrs.dat", "rb") as rf:
        attrs = pickle.load(rf)

    # Add zero row for <pad> item
    pad_row = np.zeros((1, attrs.shape[1]), dtype=np.float32)
    attrs = np.concatenate((pad_row, attrs.astype(np.float32)), axis=0)
    return attrs


def load_profiles(dataset_name):
    user_ids, item_ids = set(), set()
    profiles = defaultdict(list)

    with open(f"../data/{dataset_name}_sorted.txt", "r") as df:
        for line in df:
            user_id, item_id = list(map(int, line.strip().split(" ")))
            user_ids.add(user_id)
            item_ids.add(item_id)
            profiles[user_id].append(item_id)

    return list(user_ids), list(item_ids), profiles


def one_out_idx(profile: List[int], mode: str) -> int:
    if mode not in ["train", "val", "test"]:
        raise ValueError(f"Invalid mode: {mode}")

    if mode == "train" and len(profile) > 1:
        return max(1, len(profile) - 3)

    if mode == "val" and len(profile) > 2:
        return max(2, len(profile) - 2)

    if mode == "test" and len(profile) > 3:
        return len(profile) - 1

    return -1


def pad_profile(profile: List[int], max_len: int, mode: str) -> List[int]:
    if mode not in ["train", "val", "test"]:
        raise ValueError(f"Invalid mode: {mode}")

    start, end = 0, 0

    if mode == "train" and len(profile) > 1:
        n_excluded = 3
        start = max(0, len(profile) - n_excluded - max_len)
        end = max(1, len(profile) - n_excluded)

    if mode == "val" and len(profile) > 2:
        n_excluded = 1 if len(profile) == 3 else 2
        start = max(0, len(profile) - n_excluded - max_len)
        end = max(1, len(profile) - n_excluded)

    if mode == "test" and len(profile) > 3:
        n_excluded = 2
        start = max(0, len(profile) - n_excluded - max_len)
        end = max(1, len(profile) - n_excluded)

    return list(range(start, end))


def sample_negatives(profile: List[int], n_items: int, n: int) -> List[int]:
    sample = []

    while len(sample) < n:
        item_id = random.randint(1, n_items - 1)

        if item_id not in sample and item_id not in profile:
            sample.append(item_id)

    return sample


def get_train_sequences(
    user_id: int,
    profile: List[int],
    seq_len: int,
    attrs: np.ndarray,
    ctx: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[np.ndarray, ...]:
    q_len = attrs.shape[1] + next(iter(ctx.values())).shape[0]

    p_x = np.zeros(seq_len, dtype=np.int32)
    o_x = np.zeros(seq_len * 2, dtype=np.int32)
    p_q = np.zeros((seq_len, q_len), dtype=np.float32)
    o_q = np.zeros((seq_len * 2, q_len), dtype=np.float32)

    padded_idxs = pad_profile(profile, seq_len, "train")
    neg_sample = sample_negatives(profile, attrs.shape[0], len(padded_idxs))

    for i, pi in enumerate(padded_idxs):
        shift = seq_len - len(padded_idxs)

        p_x[shift + i] = profile[pi]
        o_x[shift + i] = profile[pi + 1]
        o_x[seq_len + shift + i] = neg_sample[i]

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_q[shift + i] = np.concatenate((a, c))

        a = attrs[profile[pi + 1]]
        c = ctx[(user_id, profile[pi + 1])]
        o_q[shift + i] = np.concatenate((a, c))

        a = attrs[neg_sample[i]]
        # Assign same context to negative sample as to positive sample
        c = ctx[(user_id, profile[pi + 1])]
        o_q[seq_len + shift + i] = np.concatenate((a, c))

    y_true = np.zeros(seq_len * 2, dtype=np.int32)
    y_true[np.where(p_x > 0)] = 1
    mask = np.zeros(seq_len * 2, dtype=np.int32)
    mask[np.where(o_x > 0)] = 1

    return p_x, p_q, o_x, o_q, y_true, mask


def get_test_sequences(
    user_id: int,
    profile: List[int],
    profile_seq_len: int,
    target_seq_len: int,
    attrs: np.ndarray,
    ctx: Dict[Tuple[int, int], np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, ...]:
    q_len = attrs.shape[1] + next(iter(ctx.values())).shape[0]

    p_x = np.zeros(profile_seq_len, dtype=np.int32)
    o_x = np.zeros(target_seq_len + 1, dtype=np.int32)
    p_q = np.zeros((profile_seq_len, q_len), dtype=np.float32)
    o_q = np.zeros((target_seq_len + 1, q_len), dtype=np.float32)

    one_out = one_out_idx(profile, mode)
    a = attrs[profile[one_out]]
    c = ctx[(user_id, profile[one_out])]
    o_x[0] = profile[one_out]
    o_q[0] = np.concatenate((a, c))

    padded_idxs = pad_profile(profile, profile_seq_len, mode)
    neg_samples = sample_negatives(profile, attrs.shape[0], target_seq_len)

    for i, pi in enumerate(padded_idxs):
        shift = profile_seq_len - len(padded_idxs)

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_x[shift + i] = profile[pi]
        p_q[shift + i] = np.concatenate((a, c))

    for i, oi in enumerate(neg_samples, start=1):
        a = attrs[oi]
        # Assign same context to negatives as to one-out positive
        c = ctx[(user_id, profile[one_out])]
        o_x[i] = oi
        o_q[i] = np.concatenate((a, c))

    y_true = np.zeros(target_seq_len + 1, dtype=np.int32)
    y_true[0] = 1
    mask = np.ones(target_seq_len + 1, dtype=np.int32)

    return p_x, p_q, o_x, o_q, y_true, mask


def get_sequences(
    user_id: int,
    profile: List[int],
    profile_seq_len: int,
    target_seq_len: int,
    attrs: np.ndarray,
    ctx: Dict[Tuple[int, int], np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, ...]:
    if mode == "train":
        return get_train_sequences(user_id, profile, profile_seq_len, attrs, ctx)
    else:
        return get_test_sequences(
            user_id, profile, profile_seq_len, target_seq_len, attrs, ctx, mode
        )


class CARCADataset(Dataset):
    def __init__(
        self,
        user_ids: List[int],
        item_ids: List[int],
        profiles: Dict[int, List[int]],
        attrs: np.ndarray,
        ctx: Dict[Tuple[int, int], np.ndarray],
        profile_seq_len: int,
        target_seq_len: int,
        mode: str,
    ):
        super().__init__()

        self.user_ids = self.valid_user_ids(profiles, mode)
        self.item_ids = item_ids
        self.profiles = profiles
        self.attrs = attrs
        self.ctx = ctx
        self.profile_seq_len = profile_seq_len
        self.target_seq_len = target_seq_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx) -> Tuple[np.ndarray, ...]:
        user_id = self.user_ids[idx]
        profile = self.profiles[user_id]

        return get_sequences(
            user_id,
            profile,
            self.profile_seq_len,
            self.target_seq_len,
            self.attrs,
            self.ctx,
            self.mode,
        )

    def valid_user_ids(self, profiles: Dict[int, List[int]], mode: str) -> List[int]:
        return [uid for uid, profile in profiles.items() if one_out_idx(profile, mode) != -1]