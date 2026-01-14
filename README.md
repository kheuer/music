# Music Genre Classification

For this project, Python 3.10 is required.

To setup a python environment, install anaconda, navigate to the projects directory and execute:

```
conda create --name music python=3.10
conda activate music
pip install -r requirements.txt
```

During development we will add dependencies to get these, simply re-run
```
pip install -r requirements.txt
```

When using Windows you can execute the following lines to download the files (you need Git Bash installed for this solution):
```
import subprocess
bash_path = r"C:\Program Files\Git\git-bash.exe" # installation path of git-bash.exe
sh_file = r"...\music\data\download.sh"
```

If there is an issue during loading all the files (NoBackendError from librosa):
```
conda install -c conda-forge ffmpeg audioread -y
```