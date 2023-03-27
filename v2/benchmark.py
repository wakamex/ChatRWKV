########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import sys
import json
import math
import time
import contextlib

import numpy as np
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_path}/../rwkv_pip_package/src")
with contextlib.suppress(Exception):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"{current_path}/../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc["text"].rsplit(" ", 1)[0], " " + doc["text"].rsplit(" ", 1)[1]] for doc in todo]

########################################################################################################

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
# MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth" # 14Btest1050
# MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-7B-Instruct-test4-20230326.pth"  # 7Btest4
MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-7B-Instruct-test4-20230326_fp16.pth"  # 7Btest4_fp16
# MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-7B-Instruct-test3-20230325.pth"  # 7Btest3
# MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-7B-Instruct-test3-20230325_fp16.pth"  # 7Btest3_fp16
# MODEL_NAME = "/data/BlinkDL/RWKV-4-Pile-7B-Instruct-test3-20230325_fp16i8.pth"  # 7Btest3_fp16i8
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = [187]

########################################################################################################

print("\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV")

torch.backends.cudnn.benchmark = True  # type: ignore
torch.backends.cudnn.allow_tf32 = True  # type: ignore
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f"Loading model - {MODEL_NAME}")

########################################################################################################

# 7Btest4      n=5153 ppl=4.35 acc=66.64 speed=0.04073693237667378 (MB used = 16_413 - 1166 = 15_247)
# 7Btest4_fp16 n=5153 ppl=4.35 acc=66.64 speed=0.04144331241218708 (MB used = 16_358 - 1110 = 15_248)
# 7Btest3      n=2000 ppl=4.17 acc=67.95 speed=0.03787282270750000 (MB used = 16_101 -  886 = 15_215)
# 7Btest3_fp16 n=5153 ppl=4.28 acc=67.07 speed=0.03784431224975742 (MB used = 16_283 - 1073 = 15_210)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16")

# 7Btest3 n=70 ppl=4.1 acc=65.71 speed=0.5916178028428571
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *0+")

# 7Btest3 n=100 ppl=4.6 acc=65.0 speed=0.43389775818
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *10+")

# 7Btest3 doesn't run
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cuda fp32 *10+")
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *30 -> cuda fp32 *0+")
# model = RWKV(model=MODEL_NAME, strategy="cuda fp32 *10+")
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *23 -> cuda fp32 *0+")
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *23 -> cuda fp32 *0+ -> cuda fp16 *1")

# 7Btest4 n=5153 ppl=4.35 acc=66.66 speed=0.04188912068464972 (MB used = 16676 - 1073 = 15_603)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *32 -> cuda fp32 *0+")

# 7Btest4 n=5153 ppl=4.35 acc=66.66 speed=0.04241702760624879 (MB used = 16648 - 1045 = 15_603)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 *32 -> cuda fp32 *1")

model = RWKV(model=MODEL_NAME, strategy="cuda fp32 *0+")

# 7Btest3 n=1000 ppl=4.29 acc=67.1 speed=0.043735098749 (MB used = 16644 - 1088 = 15556)
# 7Btest3 n=3000 ppl=4.28 acc=67.23 speed=0.03852334132333333 (MB used = 16606 - 1003 = 15603)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cuda fp32 *1")

# 7Btest3 n=630 ppl=4.22 acc=67.14 speed=0.2244852413920635 (MB used = 15754 - 1013 = 14741)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cpu fp32 *1")

# 7Btest3_fp16i8 n=50 ppl=5.19 acc=62.0 speed=1.1939671835 (MB used = 8944 - 1062 = 7882)
# 14Btest1050 n=150 ppl=4.1 acc=70.67 speed=2.3742486898466666
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16i8")

# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *0+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *10+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *1 -> cuda fp16')

# 14Btest1050 n=150 ppl=4.1 acc=71.3 speed=1.23725974588
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cuda fp16i8 *20")

# 14Btest1050 n=300 ppl=3.88 acc=72.0 speed=1.1271492735933333
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cuda fp16i8 *19 -> cuda fp16")

# 14Btest1050 n=3590 ppl=3.84 acc=70.84 speed=1.0047915335693594 (MB used = 23481 - 870 = 22611)
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16 -> cuda fp16i8 *17 -> cuda fp16")

# 14Btest1050 n=100 ppl=4.27 acc=69.0 speed=1.17113113277
# 14Btest1050 n=150 ppl=4.02 acc=71.3 speed=1.2000412033066665
# 14Btest1050 n=200 ppl=3.75 acc=73.5 speed=1.18234537701
# 14Btest1050 n=300 ppl=3.87 acc=72.0 speed=1.1814055428433332
# model = RWKV(model=MODEL_NAME, strategy="cuda fp16i8 *20 -> cuda fp16")

# 14Btest1050 didn't run
# 7btest4 didn't run
# model = RWKV(model=MODEL_NAME, strategy="cpu fp32 *20 -> cuda fp16")

# 14Btest1050 n=100 ppl=4.37 acc=70.0 speed=2.464
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 -> cpu fp32 *1')

# 14Btest1050 didn't run
# model = RWKV(model=MODEL_NAME, strategy='cpu fp32')

# 14Btest1050 didn't run
# model = RWKV(model=MODEL_NAME, strategy='cpu fp32i8')

# 14Btest1050 didn't run
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *10 -> cuda fp16 *0+')

###########################################################################
pipeline = PIPELINE(model, "20B_tokenizer.json")

print(f"Check LAMBADA on {MODEL_NAME}...{len(todo)=} samples")
xsum, xcnt, xacc = 0, 0, 0
time_ref = time.time_ns()
for d in todo:
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    logits = 0
    correct = True
    out, model_state = model.forward(src + dst, None, full_output=True)
    for i in range(len(dst)):
        probs = F.softmax(out[len(src) - 1 + i, :], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 10 == 0 or xcnt == len(todo):
        print(
            f"n={xcnt} ppl={round(math.exp(-xsum / xcnt), 2)}"
            f" acc={round(xacc / xcnt * 100, 2)}"
            f" speed={(time.time_ns()-time_ref)/xcnt/1e9}"
        )
