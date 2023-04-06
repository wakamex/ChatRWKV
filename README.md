### Getting started

```
pip install -r requirements.txt
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade
```

Download the model you want to run from Huggingface: [BlinkDL](https://huggingface.co/BlinkDL):

- latest 7B with instruct-tuning: [7B-Instruct-test4-20230326](https://huggingface.co/BlinkDL/rwkv-4-pile-7b/blob/main/RWKV-4-Pile-7B-Instruct-test4-20230326.pth)
- latest 14B: [14B-20230313-ctx8192-test1050](https://huggingface.co/BlinkDL/rwkv-4-pile-14b/blob/main/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth)

Make sure you meet the [System Requirements](#system-requirements)

Enter the path to the downloaded file in [line 48 of instruct.py](https://github.com/wakamex/ChatRWKV/blob/f827880109841d5beb77d20b35b969eedf7a2ebf/v2/instruct.py#L48)

```
python instruct.py
```

#### System Requirements

Make sure you have enough VRAM on your graphics card to run these.

- 7B requires 16GB for fp16 and 8GB for fp16i8 inference
- 14B requires 28GB for fp16 and 14GB for fp16i8 inference ([issue 30: low RAM usage](https://github.com/BlinkDL/ChatRWKV/issues/30))

Make sure you have enough CPU RAM to load these models

- 7B works with 32GB of RAM
- 14B requires 40GB+ of RAM [issue 41: not enough memory](https://github.com/BlinkDL/ChatRWKV/issues/41)

for a description of inference strategies see [https://pypi.org/project/rwkv/](https://pypi.org/project/rwkv/)
