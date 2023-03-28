import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

DATA_PATH = "../../data"


def load_ctx(dataset_name) -> Dict[Tuple[int, int], np.ndarray]:
    with open(f"{DATA_PATH}/{dataset_name}_ctx.dat", "rb") as rf:
        ctx = pickle.load(rf)

    # Cast context values from list to numpy array
    for k in ctx.keys():
        ctx[k] = np.array(ctx[k], dtype=np.float32)

    return ctx


def load_attrs(dataset_name) -> np.ndarray:
    with open(f"{DATA_PATH}/{dataset_name}_attrs.dat", "rb") as rf:
        attrs = pickle.load(rf)

    # Add zero row for <pad> item
    pad_row = np.zeros((1, attrs.shape[1]), dtype=np.float32)
    attrs = np.concatenate((pad_row, attrs.astype(np.float32)), axis=0)
    return attrs


def load_profiles(dataset_name):
    user_ids, item_ids = set(), set()
    profiles = defaultdict(list)

    with open(f"{DATA_PATH}/{dataset_name}_sorted.txt", "r") as df:
        for line in df:
            values = line.strip().split(" ")
            user_id, item_id = int(values[0]), int(values[1])
            user_ids.add(user_id)
            item_ids.add(item_id)
            profiles[user_id].append(item_id)

    return list(user_ids), list(item_ids), profiles


# def pad_profile(profile: List[int], max_len: int, mode: str) -> List[int]:
#     if mode not in ["train", "val", "test"]:
#         raise ValueError(f"Invalid mode: {mode}")
#
#     start, end = 0, 0
#
#     if len(profile) <= 3:
#         if mode == "train" and len(profile) == 2:
#             start, end = 0, 2
#
#         if mode == "val" and len(profile) == 3:
#             start, end = 0, 2
#
#         if mode == "test" and len(profile) == 3:
#             start, end = 0, 3
#     else:
#         if mode == "train" and len(profile) > 1:
#             n_excluded = 2
#             start = max(0, len(profile) - n_excluded - max_len - 1)
#             end = max(1, len(profile) - n_excluded)
#
#         if mode == "val" and len(profile) > 2:
#             n_excluded = 1
#             start = max(0, len(profile) - n_excluded - max_len - 1)
#             end = max(1, len(profile) - n_excluded)
#
#         if mode == "test" and len(profile) > 3:
#             n_excluded = 0
#             start = max(0, len(profile) - n_excluded - max_len - 1)
#             end = max(1, len(profile) - n_excluded)
#
#     return list(range(start, end))


def pad_profile(profile: List[int], max_len: int, mode: str) -> List[int]:
    if mode not in ["train", "val"]:
        raise ValueError(f"Invalid mode: {mode}")

    start, end = 0, 0

    if mode == "train" and len(profile) > 1:
        n_excluded = 1
        start = max(0, len(profile) - n_excluded - max_len - 1)
        end = max(2, len(profile) - n_excluded)

    if mode == "val" and len(profile) > 2:
        n_excluded = 0
        start = max(0, len(profile) - n_excluded - max_len - 1)
        end = max(3, len(profile) - n_excluded)

    return list(range(start, end))


def sample_negatives(profile: List[int], n_items: int, n: int) -> List[int]:
    sample = set()
    p_set = set(profile)

    while len(sample) < n:
        item_id = random.randint(1, n_items - 1)

        if item_id not in sample and item_id not in p_set:
            sample.add(item_id)

    return list(sample)


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

    for i, pi in enumerate(reversed(padded_idxs[:-1])):
        idx = seq_len - i - 1

        p_x[idx] = profile[pi]
        o_x[idx] = profile[pi + 1]
        o_x[seq_len + idx] = neg_sample[i]

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_q[idx] = np.concatenate((a, c))

        a = attrs[profile[pi + 1]]
        c = ctx[(user_id, profile[pi + 1])]
        o_q[idx] = np.concatenate((a, c))

        a = attrs[neg_sample[i]]
        # Assign same context to negative sample as to positive sample
        c = ctx[(user_id, profile[pi + 1])]
        o_q[seq_len + idx] = np.concatenate((a, c))

    y_true = np.zeros(seq_len * 2, dtype=np.int32)
    y_true[np.where(p_x > 0)] = 1

    return p_x, p_q, o_x, o_q, y_true


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

    padded_idxs = pad_profile(profile, profile_seq_len, mode)
    neg_samples = sample_negatives(profile, attrs.shape[0], target_seq_len)

    one_out = padded_idxs[-1]
    a = attrs[profile[one_out]]
    c = ctx[(user_id, profile[one_out])]
    o_x[0] = profile[one_out]
    o_q[0] = np.concatenate((a, c))

    for i, pi in enumerate(reversed(padded_idxs[:-1])):
        idx = profile_seq_len - i - 1

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_x[idx] = profile[pi]
        p_q[idx] = np.concatenate((a, c))

    for i, oi in enumerate(neg_samples, start=1):
        a = attrs[oi]
        # Assign same context to negatives as to one-out positive
        c = ctx[(user_id, profile[one_out])]
        o_x[i] = oi
        o_q[i] = np.concatenate((a, c))

    y_true = np.zeros(target_seq_len + 1, dtype=np.int32)
    y_true[0] = 1

    return p_x, p_q, o_x, o_q, y_true


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

        self.user_ids = self.valid_user_ids(profiles, profile_seq_len, mode)
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

    def valid_user_ids(self, profiles: Dict[int, List[int]], seq_len: int, mode: str) -> List[int]:
        return [uid for uid, profile in profiles.items() if pad_profile(profile, seq_len, mode)]
