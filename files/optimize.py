"""SLOTHY + Neoverse V2 最小可用模板。

執行:
    source venv/bin/activate
    export PYTHONPATH=$PWD/slothy
    python0 optimize.py gf2_kernel.s
"""

import logging
import sys

from slothy import Slothy
import slothy.targets.aarch64.aarch64_neon as AArch64_Neon
import slothy.targets.aarch64.neoverse_n1_experimental as Target_N1

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

src = sys.argv[1] if len(sys.argv) > 1 else "gf2_kernel.s"
dst = src.replace(".s", "_opt.s")

slothy = Slothy(AArch64_Neon, Target_N1)
slothy.load_source_from_file(src)

# --- 求解設定 ---
slothy.config.variable_size = True  # 讓 SLOTHY 自己找最小 cycle 數
slothy.config.timeout = 300  # 秒；大 kernel 要調高

# --- EOR3 fusion ---
# V2 上 EOR3 走 V0 (tput 1)，普通 EOR 走 V (tput 4)，融合是淨虧損。
from slothy.targets.aarch64.aarch64_neon import veor

veor.global_fusion_cb = lambda inst, t, log=None: False

# --- 上游 bug workaround ---
# 找不到 llvm-mc/llvm-nm/llvm-readobj 時，config.selftest 的 getter 會用到
# 尚未初始化的 logger 而丟 AttributeError。裝了 LLVM 就可以刪掉這行。
type(slothy.config).selftest = property(lambda self: False)

# --- software pipelining（處理迴圈時才需要）---
# slothy.config.sw_pipelining.enabled = True
# slothy.optimize_loop("loop_label")

slothy.optimize(start="slothy_start", end="slothy_end")
slothy.write_source_to_file(dst)
print(f"寫出 {dst}")
