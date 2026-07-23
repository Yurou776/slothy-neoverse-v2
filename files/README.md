# 實驗 Neoverse V2 model

## 環境建置

```bash
bash setup.sh
source venv/bin/activate
export PYTHONPATH=$PWD/slothy
```

依賴只有三個真正需要的：`ortools==9.15.6755`、`sympy==1.14.0`、`unicorn==2.1.4`。
`requirements.txt` 裡的 black / flake8 / pydoclint / sphinx 是開發用的，跑最佳化不需要。

**強烈建議額外裝 LLVM**（`llvm-mc`、`llvm-nm`、`llvm-readobj`）。SLOTHY 的
selftest 會用 unicorn 模擬器比對最佳化前後的行為是否等價——這是唯一能抓到
「排程改壞了語意」的機制。沒裝的話 selftest 直接跳過，等於裸奔。

> **上游 bug**：找不到 LLVM 時，`config.selftest` 的 getter 會呼叫尚未初始化的
> `self.logger` 而丟 `AttributeError: 'NoneType' object has no attribute 'warning'`。
> 模板裡的 `type(slothy.config).selftest = property(lambda self: False)` 是繞過方法，
> 裝了 LLVM 之後就該刪掉。

## 三個測試 kernel

| 檔案 | 測什麼 | 已驗證結果 |
|---|---|---|
| `gf2_kernel.s` | PMULL + EOR 鏈（延遲受限） | 20 cycles，IPC 0.80，selfcheck OK |
| `tput.s` | 16 條獨立 EOR + STP（吞吐受限） | **V2 7 cycles vs N1 13 cycles** |
| `scalar_kernel.s` | 43 條 scalar ALU/乘法/條件指令 | 13 cycles，IPC 3.31，無 key 衝突 |

`scalar_kernel.s` 是最重要的那個——它密集使用 flag-setting 與非 flag-setting
的 ALU 指令，專門用來測 model 裡四組 callable key
（`_alu_basic` / `_logical_basic` / `_cond_select` / `_mul_*`）會不會撞在一起。

## 執行

```bash
python3 optimize.py gf2_kernel.s      # 單一 target
python3 compare_targets.py tput.s     # A72 / N1 / V2 對照
```

## 判讀輸出

輸出檔開頭的註解區塊是關鍵：

```
// Instructions:    16
// Expected cycles: 20      ← model 預估的 cycle 數
// Expected IPC:    0.80
// Cycle bound:     20.0    ← 與 Expected 相同表示 OPTIMAL
```

`Cycle bound == Expected cycles` 代表 CP-SAT 證明了最佳性；若 bound 較小表示
timeout 了，只拿到可行解，要調高 `config.timeout`。

log 裡的 `selfcheck: OK!` 是 SLOTHY 自己的 dataflow 等價性檢查（一定會跑）；
`selftest` 才是 unicorn 模擬比對（需要 LLVM）。

## 兩種錯誤訊息的意思

| 訊息 | 意思 | 修哪裡 |
|---|---|---|
| `UnknownInstruction: Couldn't find ...` | 微架構模型少了 entry | `neoverse_v2.py` 三個 dict |
| `Multiple matches found` | 微架構模型的 key 重疊 | `neoverse_v2.py`，通常是父類與 callable 並存 |
| `ParsingException: Couldn't parse ...` | **架構**模型不支援該指令變體 | `aarch64_neon.py`，要加新 class |

第三種跟 V2 model 無關。實測時碰到的例子：`ldr w8, [x1, #8]`（W-form LDR）和
`ands x15, x11, x12`（register 形式的 ANDS）上游都沒有，得自己加 class。

## 驗證 model 準不準

SLOTHY 的 cycle 數是**模型預估**，不是實測。要確認模型可信，建議三層驗證：

1. **交叉比對 llvm-mca**
   ```bash
   llvm-mca -mcpu=neoverse-v2 -iterations=100 gf2_kernel_opt.s
   ```
   `neoverse_v2.py` 裡已經設好 `llvm_mca_target = "neoverse-v2"`。
   兩個獨立來源的估計若差很多，通常是我的 model 某一列抄錯。

2. **實機量測**（Graviton3 就是 Neoverse V2）
   包一層 loop 跑幾百萬次，用 `perf stat -e cycles,instructions` 量。
   對照 SLOTHY 的 Expected cycles × 迭代數。

3. **先測簡單的**
   從單一指令類別開始（例如只有 PMULL 的迴圈、只有 EOR 的迴圈），
   確認吞吐上限符合 SWOG 表格，再測混合 kernel。

## 已知落差（會讓預估偏樂觀）

- **§4.7 region forwarding 沒模**：跨 forwarding region 的 producer→consumer
  實際要多 1 cycle。ASIMD integer mul/mac **不屬於任何 region**，所以
  `Vmul → Vmla` 這種鏈會被低估。
- **structure load 的 `L, V`** 只模了 `L`，V port 佔用沒算。
- **post/pre-index 的 writeback µOP**（SWOG 標的那個 `I`）沒算。

這三處在 latency-bound 的 kernel 影響較小，在 throughput-bound 的混合 kernel
影響較大。實機量到系統性偏差時，優先從第一項開始補。
