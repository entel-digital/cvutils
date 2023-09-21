## Build on apple silicon

## Be aware
Just as precaution, this was done in new conda empty (not even python) environment. As far I understand, brew should not mess with conda, still didn't want to find out. I you find out, for better or worse, please document on this file.

## Setup

```bash
# install opencv from brew
brew install opencv

# find where python got installed
brew --prefix python



# install cvutils module
cd /cvutils/path
/opt/homebrew/opt/python@3.11/bin/pip3 install -e .
```


### run

```bash
/opt/homebrew/opt/python@3.11/bin/python3 -m saleskit.cli test --source="rtsp://user:pass@ip:554/path"
```

## improvements

[Maybe can python be linked?](https://docs.brew.sh/Manpage#pyenv-sync)
