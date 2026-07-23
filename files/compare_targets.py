"""同一段 code 在不同 microarchitecture model 下的 cycle 數對照。

用途: 驗證 V2 model 的行為是否合理（例如吞吐受限的 NEON code
      應該明顯優於 N1，因為 FP/ASIMD pipe 從 2 條變 4 條）。

注意: 每個 target model 支援的指令集合不同，而且都不完整。
      例如 Cortex-A55 和 A72 的 model 都缺 Q-form 的 STP
      (`stp q0, q1, [x0]`)，碰到就會丟 UnknownInstruction。
      這是那些 model 的缺口，不代表你的 code 有問題，
      所以下面對每個 target 個別 try/except，缺的就跳過。

用法:
    python3 compare_targets.py [kernel.s] [target1 target2 ...]

    python3 compare_targets.py tput.s              # 預設 N1 + V2
    python3 compare_targets.py tput.s a55 a72 n1 v2
"""

import logging
import sys

from slothy import Slothy
import slothy.targets.aarch64.aarch64_neon as A
import slothy.targets.aarch64.cortex_a55 as A55
import slothy.targets.aarch64.cortex_a72_frontend as A72
import slothy.targets.aarch64.neoverse_n1_experimental as N1
import slothy.targets.aarch64.neoverse_v2 as V2

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

ALL_TARGETS = {"a55": A55, "a72": A72, "n1": N1, "v2": V2}
# 預設只比 N1 和 V2: 兩者都是 out-of-order Neoverse, 指令覆蓋度也最接近,
# 比起來才有意義。A55/A72 主要是拿來看模型差異, 指令常常不支援。
DEFAULT = ["n1", "v2"]

src = sys.argv[1] if len(sys.argv) > 1 else "tput.s"
want = [t.lower() for t in sys.argv[2:]] or DEFAULT

for name in want:
    tgt = ALL_TARGETS.get(name)
    if tgt is None:
        print(f"=== {name.upper()}: 未知的 target, 可選: {', '.join(ALL_TARGETS)}")
        continue

    try:
        s = Slothy(A, tgt)
        s.load_source_from_file(src)
        s.config.variable_size = True
        s.config.timeout = 120
        # 上游 bug workaround: 找不到 LLVM 時 config.selftest 的 getter
        # 會用到尚未初始化的 logger。裝了 LLVM 就可以刪掉這行。
        type(s.config).selftest = property(lambda self: False)
        s.optimize(start="slothy_start", end="slothy_end")

        out = f"{src.replace('.s', '')}_{name}.s"
        s.write_source_to_file(out)
        hdr = [
            ln.strip()
            for ln in open(out)
            if "Expected cycles" in ln or "Expected IPC" in ln
        ]
        print(f"=== {name.upper()} (issue_rate={tgt.issue_rate}) -> {out}")
        for ln in hdr:
            print("   ", ln)

    except Exception as e:  # noqa: BLE001
        # 最常見的是 UnknownInstruction: 該 target model 沒有這條指令的 entry。
        # 注意看 traceback 裡是哪一個 target 的檔案在丟錯。
        print(f"=== {name.upper()} (issue_rate={tgt.issue_rate}): 跳過")
        print(f"    {type(e).__name__}: {str(e)[:140]}")
