"""
删除-保留验证门控。
"""
import torch
from src.evaluate import evaluate_model, compute_background_gap_real


def validate_candidate(
    model_original, model_candidate, test_loader, group_loaders, device='cpu',
    tau_D=0.05, tau_P=0.02,
):
    """验证候选模型是否通过删除-保留门控。

    输出：
        passed: bool
        metrics: dict
    """
    model_original.to(device)
    model_candidate.to(device)

    oa_orig, wg_orig, _ = evaluate_model(model_original, test_loader, device)
    bg_orig, flip_orig = compute_background_gap_real(model_original, group_loaders, device)

    oa_cand, wg_cand, _ = evaluate_model(model_candidate, test_loader, device)
    bg_cand, flip_cand = compute_background_gap_real(model_candidate, group_loaders, device)

    bg_drop = bg_orig - bg_cand
    oa_drop = oa_orig - oa_cand
    flip_drop = flip_orig - flip_cand

    deletion_pass = bg_drop >= tau_D
    preservation_pass = oa_drop <= tau_P
    passed = deletion_pass and preservation_pass

    return passed, {
        'bg_orig': bg_orig, 'bg_cand': bg_cand, 'bg_drop': bg_drop,
        'flip_orig': flip_orig, 'flip_cand': flip_cand, 'flip_drop': flip_drop,
        'oa_orig': oa_orig, 'oa_cand': oa_cand, 'oa_drop': oa_drop,
        'deletion_pass': deletion_pass, 'preservation_pass': preservation_pass,
    }
