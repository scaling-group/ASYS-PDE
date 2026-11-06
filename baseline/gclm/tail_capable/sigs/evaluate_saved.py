"""One common assessment of an unchanged saved single-a object."""
import argparse
import json
import os
import resource
import time
from pathlib import Path


class Deadline:
    def __init__(self,seconds):self.end=time.monotonic()+seconds
    def check(self):
        if time.monotonic()>=self.end:raise TimeoutError('Common measurement ceiling')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('saved',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--seconds',type=float,default=120)
    args=parser.parse_args()
    import torch
    from common_single import assess_family
    from single_anchor import load_profile
    torch.set_num_threads(1)
    model=load_profile(args.saved)
    cfg=json.loads((Path(__file__).parent/'assessment_config.json').read_text())
    started=time.monotonic();r=assess_family(model,anchors=(.25,),config=cfg,budget=Deadline(args.seconds))
    y=torch.logspace(-12,12,101,dtype=torch.float64)
    with torch.no_grad():predictions=model(.25,y).tolist();beta=float(model.beta(.25))
    result=dict(assessment=r,predictions=predictions,beta=beta,
                elapsed_seconds=time.monotonic()-started,
                resources=dict(user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                    system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
