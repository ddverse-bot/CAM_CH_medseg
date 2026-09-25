import subprocess
import sys

models=["unet","unet++","attunet"]
experiments=["baseline","ch","ch_cam","ch_ace","full"]

for model in models:
    for exp in experiments:
        cmd=[sys.executable,"train.py","--model",model,"--experiment",exp]
        print("Running:"," ".join(cmd))
        subprocess.run(cmd,check=True)
