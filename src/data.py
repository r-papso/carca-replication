import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

DATA_PATH = "../../data"


def set_datapath(path: str) -> None:
    global DATA_PATH
    DATA_PATH = path


def load_ctx(ctx_file: str) -> Dict[Tuple[int, int], np.ndarray]:
    with open(f"{DATA_PATH}/{ctx_file}", "rb") as rf:
        ctx = pickle.load(rf)

    # Cast context values from list to numpy array
    for k in ctx.keys():
        ctx[k] = np.array(ctx[k], dtype=np.float32)

    return ctx


def load_attrs(attr_file: str) -> np.ndarray:
    with open(f"{DATA_PATH}/{attr_file}", "rb") as rf:
        attrs = pickle.load(rf)

    # Add zero row for <pad> item
    pad_row = np.zeros((1, attrs.shape[1]), dtype=np.float32)
    attrs = np.concatenate((pad_row, attrs.astype(np.float32)), axis=0)
    return attrs


def load_profiles(profile_file: str):
    user_ids, item_ids = set(), set()
    profiles = defaultdict(list)

    with open(f"{DATA_PATH}/{profile_file}", "r") as df:
        for line in df:
            values = line.strip().split(" ")
            user_id, item_id = int(values[0]), int(values[1])
            user_ids.add(user_id)
            item_ids.add(item_id)
            profiles[user_id].append(item_id)

    return list(user_ids), list(item_ids), profiles


def pad_profile(profile: List[int], max_len: int, mode: str, test: bool) -> List[int]:
    if mode not in ["train", "val", "test"]:
        raise ValueError(f"Invalid mode: {mode}")

    start, end = 0, 0

    if mode == "train" and len(profile) > 1:
        n_excluded = 2 if test else 1
        start = max(0, len(profile) - n_excluded - max_len - 1)
        end = max(1, len(profile) - n_excluded)

    if mode == "val" and len(profile) > 2:
        n_excluded = 1 if test else 0
        start = max(0, len(profile) - n_excluded - max_len - 1)
        end = max(2, len(profile) - n_excluded)

    if mode == "test" and len(profile) > 3:
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
    test: bool,
) -> Tuple[np.ndarray, ...]:
    a_len, c_len = attrs.shape[1], next(iter(ctx.values())).shape[0]

    p_x = np.zeros(seq_len, dtype=np.int32)
    o_x = np.zeros(seq_len * 2, dtype=np.int32)

    p_a = np.zeros((seq_len, a_len), dtype=np.float32)
    o_a = np.zeros((seq_len * 2, a_len), dtype=np.float32)

    p_c = np.zeros((seq_len, c_len), dtype=np.float32)
    o_c = np.zeros((seq_len * 2, c_len), dtype=np.float32)

    padded_idxs = pad_profile(profile, seq_len, "train", test)
    neg_sample = sample_negatives(profile, attrs.shape[0], len(padded_idxs))

    for i, pi in enumerate(reversed(padded_idxs[:-1])):
        idx = seq_len - i - 1

        p_x[idx] = profile[pi]
        o_x[idx] = profile[pi + 1]
        o_x[seq_len + idx] = neg_sample[i]

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_a[idx] = a
        p_c[idx] = c

        a = attrs[profile[pi + 1]]
        c = ctx[(user_id, profile[pi + 1])]
        o_a[idx] = a
        o_c[idx] = c

        a = attrs[neg_sample[i]]
        c = ctx[(user_id, profile[pi + 1])]  # Assign same context to negative sample as to positive sample
        o_a[seq_len + idx] = a
        o_c[seq_len + idx] = c

    y_true = np.zeros(seq_len * 2, dtype=np.int32)
    y_true[np.where(p_x > 0)] = 1

    return p_x, p_a, p_c, o_x, o_a, o_c, y_true


def get_test_sequences(
    user_id: int,
    profile: List[int],
    profile_seq_len: int,
    target_seq_len: int,
    attrs: np.ndarray,
    ctx: Dict[Tuple[int, int], np.ndarray],
    mode: str,
    test: bool,
) -> Tuple[np.ndarray, ...]:
    a_len, c_len = attrs.shape[1], next(iter(ctx.values())).shape[0]

    p_x = np.zeros(profile_seq_len, dtype=np.int32)
    o_x = np.zeros(target_seq_len + 1, dtype=np.int32)

    p_a = np.zeros((profile_seq_len, a_len), dtype=np.float32)
    o_a = np.zeros((target_seq_len + 1, a_len), dtype=np.float32)

    p_c = np.zeros((profile_seq_len, c_len), dtype=np.float32)
    o_c = np.zeros((target_seq_len + 1, c_len), dtype=np.float32)

    padded_idxs = pad_profile(profile, profile_seq_len, mode, test)
    neg_samples = sample_negatives(profile, attrs.shape[0], target_seq_len)

    one_out = padded_idxs[-1]
    o_x[0] = profile[one_out]

    a = attrs[profile[one_out]]
    c = ctx[(user_id, profile[one_out])]
    o_a[0] = a
    o_c[0] = c

    for i, pi in enumerate(reversed(padded_idxs[:-1])):
        idx = profile_seq_len - i - 1
        p_x[idx] = profile[pi]

        a = attrs[profile[pi]]
        c = ctx[(user_id, profile[pi])]
        p_a[idx] = a
        p_c[idx] = c

    for i, oi in enumerate(neg_samples, start=1):
        o_x[i] = oi

        a = attrs[oi]
        c = ctx[(user_id, profile[one_out])]  # Assign same context to negatives as to one-out positive
        o_a[i] = a
        o_c[i] = c

    y_true = np.zeros(target_seq_len + 1, dtype=np.int32)
    y_true[0] = 1

    return p_x, p_a, p_c, o_x, o_a, o_c, y_true


def get_sequences(
    user_id: int,
    profile: List[int],
    profile_seq_len: int,
    target_seq_len: int,
    attrs: np.ndarray,
    ctx: Dict[Tuple[int, int], np.ndarray],
    mode: str,
    test: bool,
) -> Tuple[np.ndarray, ...]:
    if mode == "train":
        return get_train_sequences(user_id, profile, profile_seq_len, attrs, ctx, test)
    else:
        return get_test_sequences(user_id, profile, profile_seq_len, target_seq_len, attrs, ctx, mode, test)


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
        test: bool = True,
    ):
        super().__init__()

        self.user_ids = self.valid_user_ids(profiles, profile_seq_len, mode, test)
        self.item_ids = item_ids
        self.profiles = profiles
        self.attrs = attrs
        self.ctx = ctx
        self.profile_seq_len = profile_seq_len
        self.target_seq_len = target_seq_len
        self.mode = mode
        self.test = test

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx) -> Tuple[np.ndarray, ...]:
        user_id = self.user_ids[idx]
        profile = self.profiles[user_id]

        return get_sequences(
            user_id, profile, self.profile_seq_len, self.target_seq_len, self.attrs, self.ctx, self.mode, self.test
        )

    def valid_user_ids(self, profiles: Dict[int, List[int]], seq_len: int, mode: str, test: bool) -> List[int]:
        return [uid for uid, profile in profiles.items() if len(pad_profile(profile, seq_len, mode, test)) > 0]
