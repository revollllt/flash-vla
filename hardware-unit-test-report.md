# H100 硬件单元测试：访存与计算的测量方法与结论

> 一句话：**不要用数据手册的峰值当分母**。本 skill 把 H100(SXM5, sm90a, 132 SM, `acd_u` 分区)
> 拆成一个个"硬件单元测试"——每个 probe 只隔离一个引擎，测出这台机器真实能给的数字,
> 每条结论都带**成立区间(valid)**和**能推翻它的那一行(falsifier)**。目前覆盖两大类:
> **访存(copy engine / 原子 / 计数器)** 与 **计算(tensor core)**。

---

## 方法论(为什么这些数字能拿来做设计,而不是日志里的一个数)

| 原则 | 含义 | 反例(不这样就会错) |
|---|---|---|
| **单元测试 ≠ 基准测试** | 每条常数写清 claim / 成立区间 / 隔离条件 / **反证行** | 缺 falsifier 的数字是"演示",不是"实验" |
| **测周期,不测 FLOP/s** | 时钟不可锁(1.05–1.58 GHz),吞吐里裹着时钟 | 双 warpgroup 的 TFLOP/s 说"+20%",cycle 一看根本没动——20% 全是时钟差 |
| **probe 给的是天花板,不是预测** | 剥掉一切后测到的是引擎上限,真实 kernel 至多拿到这么多 | 差距是"待解释量",不是"常数错了" |
| **先设计决定性对** | 一对能区分两个假设的配置,其余是背景 | 等字节比"深度16×2KB vs 深度2×16KB"→ 判定按 transaction 还是 byte |
| **读编译产物** | `-Xptxas -v`:非零栈帧+零 spill = 局部内存数组藏在计时循环里 | `tma_ring.cu` 一个 runtime 索引的相位数组,让 tma.issue.warp 读高了 8% |

---

# 一、访存 (Memory)

## 测试 1 — TMA 投递速率(copy engine)

**测试代码**(`probes/memory/tma_ring.cu`,计时环的核心)。每个 (CTA, warp) 走一条**不相交**的流,
私有 depth 级环形缓冲;循环里**没有** wgmma、没有消费者 warp、没有计数器轮询、没有 `__syncthreads`;
坐标用移位和掩码算,把整数除法器挡在计时之外:

```cpp
// N 个 producer warp / CTA,每个 warp 一个 depth 深的私有 TMA 环
uint32_t ph = 0;                          // 相位位放进 1 个寄存器(而非 runtime 索引的局部数组)
for (int g = 0; g < trip; ++g) {
  const int s = g % depth;
  if (g >= depth) {                       // 环满,等这一格上一轮到达
    wait_wd(&mybar[s], (ph >> s) & 1, dbg, 1);
    ph ^= 1u << s;
  }
  const int idx = base * trip + g;
  const int32_t c0 = (idx & mask0) * step0;          // 沿最快维,纯移位/掩码
  const int32_t c1 = ((idx >> shift0) & mask1) * step1;
  if (lane0) {
    mybar[s].arrive_and_expect_tx(frame_b);
    tma_2d(&map, mypool + s*frame_b, c0, c1, &mybar[s]);   // cp.async.bulk.tensor.2d
  }
}
// ns/txn = us*1000/trip —— 一个不经过字节记账的除法,记账错了也伪造不出这个数
```

**数据表格**

单 warp 发射间隔与 frame 无关(depth 4):

| frame | 2 KB | 4 KB | 8 KB | 16 KB | 32 KB |
|---|---:|---:|---:|---:|---:|
| ns / TMA (8 CTA) | 245.7 | — | 245.7 | 247.6 | 253.8 |
| ns / TMA (32 CTA) | 245.7 | 247.4 | 251.0 | — | — |

聚合带宽只是**乘积** `n_ctas × n_warps × frame` 的函数(22 个 bin,132× 量程内谱宽 ±6.9%):

| 在途字节 | 投递带宽 | 占天花板 |
|---:|---:|---:|
| 256 KB | 978 GB/s | 32% |
| 512 KB | 1868 GB/s | 62% |
| 1.0 MiB | 2605 GB/s | 86% |
| 1.5 MiB | 2738 GB/s | 91% |
| 3.0 MiB | 2861 GB/s | 95% |
| 6.2 MiB | 2989 GB/s | 99% |

**反证行**:若受限于在途**字节**,等字节的 (深度16×2KB) 与 (深度2×16KB) 应相等;实测 246 vs 1319 GB/s(差 5.4×)→ 受限于 **transaction**。

**结论**
- **单 warp 每 ~248 ns 发一次 TMA,与 frame 大小无关**——所以"每 TMA 搬多少字节"是一等设计变量,frame 翻倍,copy 列几乎白赚一倍。
- 拷贝列预算 = `max(txns_per_warp × 248 ns, total_bytes / 3.02 TB/s)`;`txns_per_warp = K_per_CTA / BK`,所以**加 CTA 不动这个下界**(每个 CTA 仍走完整个 K),要动就靠 更大 BK / split-K / 更多 producer warp。
- **单 CTA 封顶 ~133 GB/s**(`tma.bw.cta.dram`):32 KB 描述符上限的单 warp 已达 91%,故**第二个 producer warp 只值 ~10%,第三个为零**。
- 环深 4 足够(≤16 KB frame 时 3 就够);单 TMA 上限 32 KB(SW128 下 128 B/行 × 256 行);稳态天花板 **3.02 TB/s**,不是手册的 3.35。

---

## 测试 2 — 全局原子与 gmem 计数器

**测试代码**(`probes/memory/gmem_atomic.cu`)。指令用模板参数固化(内联 PTX 取不了变量 opcode,
循环里放 switch 就把分支测进去了);地址在循环外算一次,让循环里只剩原子单元:

```cpp
template <int OP>
__global__ void rate_kernel(uint8_t* base, int n_addr_mask, int stride_b,
                            int trip, unsigned* sink) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t* p = base + (size_t)(gtid & n_addr_mask) * stride_b;  // 竞争度=threads/n_addr;放置=stride_b
  unsigned acc = 0;
  for (int i = 0; i < trip; ++i) one_op<OP>(p, 1u, acc);        // red / atom / v4.f32 / cas ...
  if (acc == 0xffffffffu) sink[gtid] = acc;                    // 永假,但保证返回值不被消除
}
// 计数器一跳(release 递增 → 另一 CTA acquire 观测)用 ping-pong 从 host 计时,
// 避免跨两个 SM 比 clock64();observer 只读不推进,分离"观测代价"与"推进代价"。
```

**数据表格**

`red.u32` 吞吐(Gop/s),33792 线程,按 **地址数 × 地址间距**(**加粗 = 该竞争度下更优放置**):

| 地址数 | 每址共享线程 | 间距 4 B | 间距 128 B |
|---:|---:|---:|---:|
| 1 | 33792 | 1.4 | 1.4 |
| 4 | 8448 | 5.4 | 2.3 |
| 32 | 1056 | 12.6 | 10.5 |
| 256 | 132 | 27.2 | **64.9** |
| 2048 | 16 | **165.4** | 70.5 |
| 16384 | 2 | **525.6** | 83.1 |

计数器一跳延迟(ns / arrive→observe):

| 推进方式 | 0 观测者 | 6 | 30 | 130 |
|---|---:|---:|---:|---:|
| `red.release.gpu.add` | 651.1 | 648.5 | 648.4 | 653.8 |
| `st.release.gpu`(裸旗标) | 564.4 | 570.4 | 574.1 | 563.9 |

**结论**
- **地址放置值 6.3×,是本单元最大的杠杆,其余 ≤1.3×**。把一个 warp 的 32 个原子塞进同一条 128 B line(少共享时)最快;重共享时反而慢 2.4×,交叉点 ~100 线程/地址。→ split-K 部分和(少 lane 共享)要**打包**,直方图(多 lane 撞同 bin)要**打散/私有化**。
- **按 transaction 计,不按 byte**:`red.global.add.v4.f32` 与 `.u32` 同速率,**白搬 3.8× 字节**(340 GB/s → 1.3 TB/s)。
- `red` 比返回旧值的 `atom` 快 **1.30×**;scope(cta/gpu/sys)**免费**,只按正确性选;单个竞争地址 1.36 Gop/s,比最优低 **386×**。
- **计数器一跳 ~650 ns,观测者免费**(130 个 CTA 轮询同一计数器仅多 0.4%)→ 一个计数器可门控整机,广播树是在解一个这台硬件不存在的问题。

---

# 二、计算 (Compute)

## 测试 3 — wgmma 发射速率(tensor core)

**测试代码**(`probes/compute/mma_rate.cu`)。一个 warpgroup 从常驻 smem 背靠背发 wgmma,
循环里无 TMA、无全局访存、无 barrier;用 in-kernel `clock64` 计**周期**(躲开不可锁的时钟);
每个 N 先过一次 torch 对拍(相对误差 5e-8~1e-7)才信这个速率:

```cpp
fence_acc(acc);                                   // warpgroup_fence_operand:少了它 ptxas 报 C7515 并串行化流水,
asm volatile("wgmma.fence.sync.aligned;");        //   循环照跑,产出的却是"串行速率"——probe 把 C7515 当编译失败
__syncthreads();
const long long t0 = clock64();
for (int i = 0; i < trip; ++i) {
  for (int j = 0; j < NGROUP; ++j) Atom{}.call(tA, tB, tC);   // 在飞条数 = NGROUP*(WAIT+1)
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(WAIT));  // WAIT=0 每轮排空流水
}
const long long t1 = clock64();
// 注:-arch=sm_90a 在本工具链会退化到 compute_90,ptxas 直接拒 wgmma.*;
//     必须 -gencode arch=compute_90a,code=sm_90a
```

**数据表格**

单指令周期 vs 输出 tile N(1 CTA 与 132 CTA 相同,是 per-SM 速率):

| N | 8 | 16 | 32 | **64** | 96 | 128 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 周期 / 指令 | 18.9 | 20.7 | 24.7 | **32.8** | 48.7 | 65.0 | 97.1 | 129.3 |
| 架构理想 | 3.8 | 7.7 | 15.3 | 30.7 | 46.0 | 61.4 | 92.1 | 122.8 |
| **占峰值** | 20% | 37% | 62% | **94%** | 95% | 94% | 95% | 95% |

流水深度(在飞条数 = 组大小 × (wait+1)),N=64,理想 30.7:

| 组大小 → | 1 | 1 | 1 | 2 | 2 | 4 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wait 深度 | 0 | 1 | 3 | 0 | 1 | 0 | 1 | 0 |
| 在飞条数 | 1 | 2 | 4 | 2 | **4** | 4 | **8** | 8 |
| 周期 / 指令 | 79.0 | 42.0 | 33.7 | 55.6 | **33.1** | 44.6 | **32.3** | 38.2 |

**结论**
- **N 必须 ≥ 32,否则别用 wgmma**:N=8 花 18.9 周期做 3.8 周期的活(5× 浪费),N=32 仍浪费 38%。N≥64 后周期与 FLOP 严格成正比,**N 免费**——取寄存器预算允许的最大值。
- **主循环里绝不写 `wgmma.wait_group 0`**(它每轮排空流水,值 20–30%,还是最顺手写出来的那种)。**4 条在飞 + wait≥1** 是拐点;这同时是寄存器决策(N=64 时每组在飞 = 32 个累加寄存器/线程)。
- **一个 warpgroup 已经喂饱 tensor core**:第二个 warpgroup 恰好让各自单指令周期翻倍(1.94–1.99×),聚合吞吐不动。加第二个数学 warpgroup 只为别的理由(累加器 > 255 寄存器 / 与 epilogue 重叠),不是为喂得更快。
- 时钟在负载下只跑 1.05–1.58 GHz,**实际 bf16 天花板 ~850 TFLOP/s**,不是手册的 989——目标定"周期",别定这个 989 派生的 FLOP/s。

---

## 测试 4 — `mma.sync`(warp 级)vs `wgmma`:交叉点

**测试代码**(同 probe)。操作数先入寄存器(NACC=1 把每条指令链到上一条结果 → 测**延迟**;
NACC>1 测**吞吐**,和 TMA 环深是同一招"用一个轴分开延迟与速率"):

```cpp
uint32_t a[4], b[2];
load_frags(gA, gB, a, b);            // m16n8k16 片段布局,写明而非藏起来,由 check_kernel 对拍验证
float d[NACC][4] = {};
const long long t0 = clock64();
for (int i = 0; i < trip; ++i)
  for (int j = 0; j < NACC; ++j)     // NACC 个独立累加器
    mma_16816(d[j][0], d[j][1], d[j][2], d[j][3],
              a[0], a[1], a[2], a[3], b[0], b[1]);  // mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32
const long long t1 = clock64();
```

**数据表格**

同一 job / 同 SM / 同时钟源,统一到 **FLOP / cycle / SM**(warp 级与 warpgroup 级唯一可比的轴):

| wgmma tile N | 8 | 16 | **32** | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| wgmma | 871 | 1577 | **2623** | 3959 | 4036 | 4053 |
| best `mma.sync` | 2680 | 2680 | **2680** | 2680 | 2680 | 2680 |
| 胜者 | sync **3.1×** | sync **1.7×** | **平** | wgmma 1.5× | wgmma 1.5× | wgmma 1.5× |

`mma.sync` 独立累加器数(延迟 ~25 cyc / 发射间隔 6.26 cyc → 恰好需 4 个):

| 独立累加器 | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| 周期 / 指令 | 25.14 | 12.54 | **6.26** | 6.29 |

**结论**
- **输出 tile N < 32 的 kernel 应该用 `mma.sync` 而非 `wgmma`,最高值 3.1×**——与"永远用最新指令"的直觉相反,而本仓库 decoder 形状恰恰是小 N。N≥64 用 wgmma 拿 1.5×。
- 但 `mma.sync` 自身天花板只有 **2680 FLOP/cyc/SM = 架构峰值的 63%**(wgmma 是 95%)——它喂不满 tensor core,交叉点以下才划算。
- **每个引擎都想要"4 件在飞"**:wgmma 4 条在飞、mma.sync 4 个累加器/warp、TMA 环深 4——一个好用的默认值,也便宜好验。
- 补充(`mma.feedtax.warp.ldmatrix`):真实主循环 mma.sync 每步要付 `ldmatrix` 税——只要每 ldmatrix ≥0.8 条 mma 就 ≤1.18×;低于此(16×8 tile,0.5)暴涨 4.46×(掉到 14% 峰值),细粒度分解最容易踩这个坑。

---

# 三、结论总表(可直接进设计预算)

| 标签 | 结论 | 设计用法 |
|---|---|---|
| `tma.issue.warp` | 248 ns/TMA/warp,与 frame 无关 | copy 列 = `max(txns×248ns, bytes/3.02TB/s)`;frame 是一等变量 |
| `tma.bw.cta.dram` | 单 CTA 封顶 133 GB/s | 第 2 个 producer warp +10%,第 3 个为零 |
| `tma.bw.dev.curve` | 带宽 = `CTA×warp×frame` 乘积的函数 | 90% 天花板:32KB frame 需 ~48 CTA;查曲线而非插值 |
| `tma.bw.dev.dram` | 稳态 3.02 TB/s(非 3.35) | 长主循环用它当分母;短 kernel 用 `1.85+MB/2.77` |
| `atom.ratio.place` | 地址放置值 6.3× | split-K 打包进 128B line;直方图打散 |
| `atom.ratio.width` | 按 transaction,`v4.f32` 白搬 3.8× | 部分和累加用最宽的原子 |
| `atom.lat.dev.hop` | 计数器一跳 650 ns,观测者免费 | 一个计数器门控整机;任务粒度须 > ~2MB 流量才值得单独排序 |
| `wgmma.issue.wg.ss` | wgmma N≥64 才 94–95% 峰值 | 小 N tile 拿不到 tensor core 峰值,是切分约束 |
| `wgmma.stages.wg.knee` | 4 条在飞 + wait≥1 | 主循环绝不 `wait_group 0`(值 20–30%) |
| `wgmma.ratio.sm.wg2` | 一个 warpgroup 已饱和 | 第 2 个数学 WG 需独立理由,不为喂得更快 |
| `mma.xover.n.wgmma` | N=32 交叉:< 32 用 mma.sync | 最高 3.1×;≥64 回到 wgmma |
| `wgmma.clock.sm` | 实际 bf16 天花板 ~850 TFLOP/s | 目标定"周期",别定 989 派生的 FLOP/s |

**方法学一句话总纲**:测这台机器,再照着它设计。凡是从数据手册峰值除出来的目标,永远不会被打破——因为它从一开始就够不着。

> 机器/工具链:H100 80GB HBM3,sm90a,`acd_u` 分区,时钟不可锁(~6% 噪声底)。
> 访存/计算单元 torch 2.13.0+cu130 / CUDA 13.1(2026-08-25);计时器 CUPTI + in-kernel `clock64`。
> 每条常数的成立区间、隔离条件、反证行与 job id 见 `sm90/constants.yaml`(用 `scripts/constants.py --tag <TAG>` 读)。
> **最大未测项**:`wgmma.bytes.wg.tma` 说一个 CTA 每数学 warpgroup 需 ~4 producer warp,但那是两个分开 kernel 测的常数做的算术——copy 引擎与 tensor core 在同一 kernel 里是否真能并发(而非争用),尚无 probe 跑过。
