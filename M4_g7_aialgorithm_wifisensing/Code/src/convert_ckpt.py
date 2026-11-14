import os, sys, torch
from models.wiflexformer_dat import WiFlexFormerDAT

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(dst), exist_ok=True)
m = WiFlexFormerDAT(grl_lambda=8.0)
m.load_state_dict(torch.load(src, map_location="cpu"))
torch.save(m, dst)
print("saved", dst)
