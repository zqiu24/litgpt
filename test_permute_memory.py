import os, psutil, torch

def check_permutation_copy(shape=(8, 1024, 4096), device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(*shape, device=device)
    perm = torch.randperm(x.size(-1), device=device)

    # Measure memory before
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    else:
        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss

    y = x[..., perm]  # the operation in question

    # Measure memory after
    if device == 'cuda':
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
    else:
        after = psutil.Process(os.getpid()).memory_info().rss

    print("is view (has base):", y._base is not None)  # Expect: False
    print("same storage ptr:", y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # Expect: False
    print("allocated delta (bytes):", after - before)  # Expect: > 0 for sufficiently large tensors

    # Mutate y in-place; x should not change if y is not a view
    y.fill_(0)
    print("x changed after y.fill_(0):", bool((x == 0).all().item()))  # Expect: False

check_permutation_copy()