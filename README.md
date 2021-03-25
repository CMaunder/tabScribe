# TabScribe

Install anaconda and add to path from webpage
in cmd:
- conda create -n tabScribeEnv python=3.8,
    then launch env (as below):
- Launch env:
    conda activate noteRecogEnv
- Deactivate env:
    conda deactivate
- See envs:
    conda info --envs
- Remove env: 
    conda remove --name myenv --all

Then: 
- ensure you have installed all CUDA requirement and run: conda install -c conda-forge tensorflow-gpu
- pip install -r requirements.txt