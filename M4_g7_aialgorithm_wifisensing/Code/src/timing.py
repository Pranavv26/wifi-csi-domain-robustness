import time, json, torch
from models.wiflexformer_dat import WiFlexFormerDAT
torch.manual_seed(0)
m = WiFlexFormerDAT(grl_lambda=8.0).eval()
params = sum(p.numel() for p in m.parameters())
out = {"params": params, "threads": torch.get_num_threads(), "device": "cpu (AMD Ryzen 5 5500U)"}
for bs in (1, 128):
    x = torch.randn(bs, 1, 30, 220)
    with torch.no_grad():
        for _ in range(5): m(x)
        n = 200 if bs == 1 else 10
        t0 = time.perf_counter()
        for _ in range(n): m(x)
        out[f"ms_per_sample_bs{bs}"] = 1000 * (time.perf_counter() - t0) / (n * bs)
print(json.dumps(out, indent=2))
json.dump(out, open("tta_results/timing_pranav_cpu.json", "w"), indent=2)
