"""
Utility functions for the EE4740 Huffman Coding miniproject.
"""

import heapq
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav


# ============================================================
# Data loading and splitting
# ============================================================


def load_files(folder):
    files = sorted(os.listdir(folder))
    data = []

    for f in files:
        sr, d = wav.read(os.path.join(folder, f))
        d = d / np.max(np.abs(d))
        data.append((f, sr, d))

    return data


# 80/20 split per gender, depending on the ratio passed in.
def split_test_ratio(files, ratio):
    n_train = int(len(files) * ratio)
    train = files[:n_train]
    test = files[n_train:]

    return train, test


def make_mixed_training_pool(female_train, male_train, seed=0):
    """
    Create one mixed training pool from all female and male training files.

    The samples are shuffled with a fixed seed for reproducibility. Huffman
    coding depends only on symbol frequencies, not on sample order, but
    shuffling avoids the pool being ordered by file or gender.
    """
    sr_f = female_train[0][1]
    sr_m = male_train[0][1]

    if sr_f != sr_m:
        raise ValueError("Female and male files have different sample rates.")

    sr = sr_f

    female_signal = np.concatenate([d for _, _, d in female_train])
    male_signal = np.concatenate([d for _, _, d in male_train])
    train_pool = np.concatenate([female_signal, male_signal])

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)

    return train_pool, sr


# ============================================================
# Quantisation and reconstruction
# ============================================================


def quantize(signal, n_levels):
    edges = np.linspace(-1, 1, n_levels + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    indices = np.digitize(signal, edges) - 1
    indices = np.clip(indices, 0, n_levels - 1)

    return indices, centres


def reconstruct_from_symbols(symbols, centres):
    symbols = np.array(symbols, dtype=int)
    symbols = np.clip(symbols, 0, len(centres) - 1)

    return centres[symbols]


# ============================================================
# Huffman coding
# ============================================================


def build_codebook(indices, n_levels):
    frequencies = Counter(indices)

    # Ensure every possible quantisation level appears in the alphabet.
    for symbol in range(n_levels):
        frequencies[symbol] += 1

    codebook = build_huffman_tree(frequencies)

    return codebook, frequencies


def build_huffman_tree(frequencies):
    heap = [[freq, [symbol, ""]] for symbol, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = "0" + pair[1]

        for pair in hi[1:]:
            pair[1] = "1" + pair[1]

        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(sorted(heapq.heappop(heap)[1:], key=lambda x: len(x[1])))


def encode(symbols, codebook):
    return "".join(codebook[s] for s in symbols)


def decode(bitstream, codebook):
    reverse = {v: k for k, v in codebook.items()}
    symbols = []
    current = ""

    for bit in bitstream:
        current += bit

        if current in reverse:
            symbols.append(reverse[current])
            current = ""

    return symbols


def encode_symbols(symbols, codebook):
    return "".join(codebook[int(s)] for s in symbols)


def decode_forward(bitstream, codebook, max_symbols=None):
    reverse = {v: k for k, v in codebook.items()}
    out = []
    buf = ""

    for bit in bitstream:
        buf += bit

        if buf in reverse:
            out.append(reverse[buf])
            buf = ""

            if max_symbols is not None and len(out) >= max_symbols:
                break

    return np.array(out, dtype=int)


def decode_prefix(bitstream, codebook, max_symbols=None):
    reverse = {v: k for k, v in codebook.items()}
    out = []
    buf = ""

    for bit in bitstream:
        buf += bit

        if buf in reverse:
            out.append(reverse[buf])
            buf = ""

            if max_symbols is not None and len(out) >= max_symbols:
                break

    return np.array(out, dtype=int)


def average_code_length(codebook, freqs):
    total = sum(freqs.values())
    return sum(freqs[s] * len(codebook[s]) for s in freqs) / total


# ============================================================
# Channel/error models
# ============================================================


def inject_errors(bitstream, error_rate):
    bits = list(bitstream)

    for i in range(len(bits)):
        if random.random() < error_rate:
            bits[i] = "1" if bits[i] == "0" else "0"

    return "".join(bits)


def lose_bits(bitstream, p_loss, rng):
    keep = rng.random(len(bitstream)) >= p_loss
    return "".join(np.array(list(bitstream))[keep])


def apply_random_bit_loss(bitstream, loss_probability, rng):
    """Randomly delete bits from the bitstream."""
    keep = rng.random(len(bitstream)) > loss_probability
    return "".join(np.array(list(bitstream))[keep])


def damage_middle(bitstream, damage_fraction):
    n = len(bitstream)
    damage_len = int(damage_fraction * n)

    if damage_len == 0:
        return bitstream, ""

    start = (n - damage_len) // 2
    end = start + damage_len

    return bitstream[:start], bitstream[end:]


# ============================================================
# Metrics
# ============================================================


def snr_db(original, reconstructed):
    n = min(len(original), len(reconstructed))
    err = original[:n] - reconstructed[:n]

    if np.mean(err**2) == 0:
        return np.inf

    return 10 * np.log10(np.mean(original[:n] ** 2) / np.mean(err**2))


def symbol_error_rate(original, decoded):
    n = min(len(original), len(decoded))

    if n == 0:
        return 1.0

    aligned_errors = np.mean(original[:n] != decoded[:n])
    length_penalty = abs(len(original) - len(decoded)) / len(original)

    return min(1.0, aligned_errors + length_penalty)


def recovered_prefix_fraction(original, decoded):
    n = min(len(original), len(decoded))
    correct = np.sum(original[:n] == decoded[:n])

    return correct / len(original)


# ============================================================
# RVLC construction and decoding
# ============================================================


def is_compatible(candidate, existing_codes):
    for code in existing_codes:
        if candidate.startswith(code) or code.startswith(candidate):
            return False

        if candidate.endswith(code) or code.endswith(candidate):
            return False

    return True


def build_rvlc_codebook(symbols, n_levels, max_len=32):
    freqs = Counter(symbols)

    for s in range(n_levels):
        freqs.setdefault(s, 1)

    total = sum(freqs.values())
    symbol_order = sorted(freqs.keys(), key=lambda s: freqs[s], reverse=True)

    probs = {s: freqs[s] / total for s in freqs}
    target_lengths = {
        s: max(2, int(np.ceil(-np.log2(probs[s]))))
        for s in freqs
    }

    codebook = {}
    existing = []

    for s in symbol_order:
        assigned = False

        for L in range(target_lengths[s], max_len + 1):
            for x in range(2**L):
                candidate = format(x, f"0{L}b")

                if is_compatible(candidate, existing):
                    codebook[s] = candidate
                    existing.append(candidate)
                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            raise RuntimeError(f"Could not assign RVLC code for symbol {s}")

    return codebook, freqs


def decode_backward(bitstream, codebook, max_symbols=None):
    reverse = {v: k for k, v in codebook.items()}
    out = []
    buf = ""

    for bit in bitstream[::-1]:
        buf = bit + buf

        if buf in reverse:
            out.append(reverse[buf])
            buf = ""

            if max_symbols is not None and len(out) >= max_symbols:
                break

    out.reverse()
    return np.array(out, dtype=int)


# ============================================================
# Robustness helpers
# ============================================================


def huffman_recover_from_damaged_block(symbols, codebook, damage_fraction):
    bitstream = encode_symbols(symbols, codebook)
    left_bits, right_bits = damage_middle(bitstream, damage_fraction)

    if damage_fraction == 0:
        return decode_forward(left_bits + right_bits, codebook)

    return decode_forward(left_bits, codebook)


def rvlc_recover_from_damaged_block(symbols, codebook, damage_fraction):
    bitstream = encode_symbols(symbols, codebook)
    left_bits, right_bits = damage_middle(bitstream, damage_fraction)

    left_decoded = decode_forward(left_bits, codebook)
    right_decoded = decode_backward(right_bits, codebook)

    return np.concatenate([left_decoded, right_decoded])


def decode_rvlc_block_forward_backward(damaged_bits, codebook):
    """
    Decode a damaged RVLC block from both ends.

    Forward decoding recovers symbols near the start. Backward decoding
    recovers symbols near the end.
    """
    fwd = decode_forward(damaged_bits, codebook)
    bwd = decode_backward(damaged_bits, codebook)

    return fwd, bwd


def huffman_block_symbol_error_rate(original_block, damaged_bits, codebook):
    """Ordinary Huffman decoding from the left only."""
    decoded = decode_forward(damaged_bits, codebook)

    n = len(original_block)
    recovered = np.full(n, -1, dtype=int)
    n_dec = min(len(decoded), n)

    if n_dec > 0:
        recovered[:n_dec] = decoded[:n_dec]

    return np.mean(recovered != original_block)


def rvlc_block_symbol_error_rate(original_block, damaged_bits, codebook):
    """
    Approximate RVLC recovery after random bit loss.

    Since the exact missing-bit positions are unknown at the decoder, this
    scores recovered prefix from forward decoding and recovered suffix from
    backward decoding.
    """
    fwd, bwd = decode_rvlc_block_forward_backward(damaged_bits, codebook)

    n = len(original_block)
    n_fwd = min(len(fwd), n)
    n_bwd = min(len(bwd), n - n_fwd)

    recovered = np.full(n, -1, dtype=int)

    if n_fwd > 0:
        recovered[:n_fwd] = fwd[:n_fwd]

    if n_bwd > 0:
        recovered[n - n_bwd:] = bwd[-n_bwd:]

    return np.mean(recovered != original_block)


# ============================================================
# Evaluation
# ============================================================


def evaluate_rvlc_bit_loss(
    symbols,
    huffman_codebook,
    rvlc_codebook,
    bit_loss_probs,
    block_sizes,
    n_blocks=100,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows = []

    symbols = np.array(symbols, dtype=int)

    for p_loss in bit_loss_probs:
        # No resynchronisation: one long Huffman stream.
        full_bits = encode_symbols(symbols, huffman_codebook)
        damaged_full_bits = apply_random_bit_loss(full_bits, p_loss, rng)
        decoded_full = decode_forward(damaged_full_bits, huffman_codebook)

        recovered_full = np.full(len(symbols), -1, dtype=int)
        n_dec = min(len(decoded_full), len(symbols))
        recovered_full[:n_dec] = decoded_full[:n_dec]

        rows.append(
            {
                "bit_loss_probability": p_loss,
                "method": "Huffman: no resynchronisation",
                "symbol_error_rate": np.mean(recovered_full != symbols),
            }
        )

        for block_size in block_sizes:
            huff_errors = []
            rvlc_errors = []

            max_start = len(symbols) - block_size

            for _ in range(n_blocks):
                start = rng.integers(0, max_start)
                block = symbols[start:start + block_size]

                huff_bits = encode_symbols(block, huffman_codebook)
                damaged_huff_bits = apply_random_bit_loss(huff_bits, p_loss, rng)

                huff_errors.append(
                    huffman_block_symbol_error_rate(
                        block,
                        damaged_huff_bits,
                        huffman_codebook,
                    )
                )

                rvlc_bits = encode_symbols(block, rvlc_codebook)
                damaged_rvlc_bits = apply_random_bit_loss(rvlc_bits, p_loss, rng)

                rvlc_errors.append(
                    rvlc_block_symbol_error_rate(
                        block,
                        damaged_rvlc_bits,
                        rvlc_codebook,
                    )
                )

            rows.append(
                {
                    "bit_loss_probability": p_loss,
                    "method": f"Huffman block size {block_size}",
                    "symbol_error_rate": np.mean(huff_errors),
                }
            )

            rows.append(
                {
                    "bit_loss_probability": p_loss,
                    "method": f"RVLC block size {block_size}",
                    "symbol_error_rate": np.mean(rvlc_errors),
                }
            )

    return pd.DataFrame(rows)
