import torch

from scripts.prepare_itop_e1_cache import _sample_ids


def test_itop_e1_sample_ids_encode_view_and_frame():
    payload = {
        "frame_index": torch.tensor([4, 4, 9]),
        "view_id": torch.tensor([0, 1, 0]),
    }
    identifiers = _sample_ids(payload)
    assert torch.unique(identifiers).numel() == 3
    assert identifiers.tolist() == [4, (1 << 32) + 4, 9]
