"""Eight fixed original-size blocks; only collocation density changes."""
from pathlib import Path
import hashlib,json
import torch
ROOT=Path(__file__).resolve().parent
BLOCKS=((107014,107015),(107024,107025),(107034,107035),(107044,107045),(107054,107055),(107064,107065),(107074,107075),(107084,107085))
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
def dense_batch(problem, blocks=8):
 assert blocks in (1,8)
 original=problem.training_batch(dict(Nf=512,NIC=64,NBC=32))
 uniform=[original[0]];core=[problem.core_points(192,BLOCKS[0][1])]
 for uniform_seed,core_seed in BLOCKS[1:blocks]:
  uniform.append(problem.batch(count=512,ic_count=64,boundary_count=32,seed=uniform_seed)[0])
  core.append(problem.core_points(192,core_seed))
 metadata=[]
 for i,((us,cs),u,c) in enumerate(zip(BLOCKS[:blocks],uniform,core)):
  assert u.shape==(512,2) and c.shape==(208,2)
  assert int((c[:,0]==0).sum())==16
  metadata.append(dict(block=i,uniform_seed=us,core_seed=cs,uniform_sha256=tensor_hash(u),core_sha256=tensor_hash(c)))
 u,c=torch.cat(uniform),torch.cat(core)
 assert torch.isfinite(u).all() and torch.isfinite(c).all()
 assert u.shape==(512*blocks,2) and c.shape==(208*blocks,2) and int((c[:,0]==0).sum())==16*blocks
 return (u,original[1],original[2]),c,dict(blocks=metadata,uniform_count=512*blocks,core_count=208*blocks,off_origin_count=192*blocks,origin_count=16*blocks,core_weight=.2,construction='fixed blocks of 512 uniform and 208 core points')

def verify_initial(model,seed):
 bindings=json.loads((ROOT/'ORIGINAL_BINDINGS.json').read_text())
 path=ROOT/'paired_initials'/f'{seed}.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
 assert digest==bindings['initials'][str(seed)]['sha256']
 expected=torch.load(path,map_location='cpu',weights_only=True);actual=model.state_dict()
 assert actual.keys()==expected.keys()
 assert all(v.dtype==torch.float64 and torch.isfinite(v).all() and torch.equal(v,expected[k]) and tensor_hash(v)==tensor_hash(expected[k]) for k,v in actual.items()),'Constructor differs from paired unfitted initial state'
 assert torch.count_nonzero(model.net[-1].weight)==0 and torch.count_nonzero(model.net[-1].bias)==0
 return digest  # No model.load_state_dict: initialization is the unchanged fresh constructor.
