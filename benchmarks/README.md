# ChromBERT 官方任务 Benchmark

以 ChromBERT 官方教程为任务定义。每种任务在 `tasks/` 中有独立的 Python 文件。`training.py` 提供共用的 Lightning 训练逻辑，`runner.py` 组织运行、计时、导出和验收。所有操作从 `benchmark.py` 进入。原版模型从上游固定提交 `10589002597a9ef06613ed41b6ee83e8dd565c25` 打包，官方 Notebook 随结果保存。

## 全量配置

命令中的任务名决定使用哪份数据划分。默认的 `--mode full` 会在选定配置上执行从数据准备到结果验收的完整流程。

| 配置 | 预测任务 | 训练 / 验证 / 测试 | 主要测试指标 |
| --- | --- | --- | --- |
| [`gep-all`](configs/gep-all.json) | 成纤维细胞向成肌细胞转换时的基因表达变化 | 15,720 / 1,965 / 1,965 个基因 | Pearson、Spearman、MSE、MAE、R² |
| [`ezh2`](configs/ezh2.json) | 人胚胎干细胞 EZH2 位点分类 | 8,806 / 1,101 / 1,101 个区间 | AUROC、AUPRC、BCE |
| [`cistrome`](configs/cistrome.json) | 用官方 prompt 权重直接推理，补全 MAZ × K562 的结合信号 | — / — / 2,137,894 个 hg38 区间 | AUROC、AUPRC、BCE |
| [`eqtl`](configs/eqtl.json) | 肺组织 eQTL 分类，按基因组区间分组划分 | 918 / 312 / 305 个变异 | AUROC、AP、BCE |
| [`starr`](configs/starr.json) | 预测 STARR-seq 的野生型和扰动后活性及其差值 | 26,208 条训练记录 / 4,082 条验证记录 / 721 个测试位点对 | 各状态及差值的 Pearson、差值 RMSE、预测方差 |

### 抽样和教程配置

| 配置 | 用途 | 训练 / 验证 / 测试 |
| --- | --- | --- |
| [`gep-sampled`](configs/gep-sampled.json) | 在 GEP 全量划分的各集合内按表达变化分层抽取 20% | 3,144 / 393 / 393 个基因 |
| [`cistrome-sampled`](configs/cistrome-sampled.json) | 按染色体和标签分层抽样，复用保存的区间清单 | — / — / 250,000 个区间 |
| [`gep`](configs/gep.json) | 复现官方教程的小样本设置，并在同一测试集评估官方微调权重 | 80 / 20 / 50 个基因 |
| [`eqtl-official`](configs/eqtl-official.json) | 复现官方教程的随机划分和训练流程 | 921 / — / 614 个变异 |

训练种子、权重选择、精度和数据校验值见各配置文件。

## 可解释性

`ezh2`、`gep-all` 和 `gep-sampled` 使用本轮测试预测所用的 checkpoint 提取调控因子嵌入。EZH2 在经典和非经典位点组内分别平均嵌入，按候选因子与 EZH2 的余弦相似度及两组差值排序。GEP 使用所选配置的测试基因，按上调和表达近似不变分组，以两组平均嵌入的余弦相似度从低到高排序。

运行输出 `regulator_rankings.csv`、用于重算排名的 `regulator_means.npz`，以及 `metrics.json` 中的样本分组和独立阶段计时。Cistrome 的在线 MAZ × K562 任务输出区间级预测概率和分类指标，其标签与概率也可用于画教程的 ROC 曲线。

## 数据准备

从仓库根目录执行命令。先将 `BENCHMARK_WORKSPACE` 设为服务器已有的共享实验工作区绝对路径。

```bash
export BENCHMARK_WORKSPACE=/absolute/path/to/shared/chrombert
```

工作区的 `assets/hg38_1kb/` 存放 HDF5、参考 BED、元数据、权重和官方任务数据。`cache/` 存放从官方 demo 提取的输入，`bundles/` 存放运行包，`runs/benchmark/` 存放结果。各实验分支使用同一个工作区和 GPU 队列锁。训练和 Profiler 串行运行，每次选择一张空闲卡。每次运行使用现有实验室镜像启动独立容器，将资产和运行包以只读方式挂载。预检核对已固定的 SHA-256，HDF5 记录体积和修改时间。

GEP 配置需要官方 `demo.tar.gz` 中的两份表达 CSV。教程演示还使用 `gep_fibroblast_to_myoblast.ckpt`。资产目录已安装官方 demo 时可直接预检。也可先下载并提取到工作区缓存，打包命令会校验输入。

```bash
mkdir -p "$BENCHMARK_WORKSPACE/cache"
curl --fail --location --retry 2 \
  --output "$BENCHMARK_WORKSPACE/cache/demo.tar.gz" \
  https://huggingface.co/datasets/TongjiZhanglab/chrombert/resolve/main/demo.tar.gz
python3 benchmarks/benchmark.py extract-gep "$BENCHMARK_WORKSPACE/cache/demo.tar.gz" \
  --destination "$BENCHMARK_WORKSPACE/cache/official-gep"
```

容器需要官方模型的运行依赖、Lightning、TensorBoard 和常用科学计算库。EZH2 与 Cistrome 还需要 BEDTools。预检会报告缺失文件或依赖以及实际 GPU 和软件版本。

可解释性阶段使用官方 `hg38_6k_factors_list.txt`。清单可放在服务器资产目录的 `config/`，也可放在共享工作区的 `cache/` 随包装入容器。预检记录清单的 SHA-256。[官方因子清单](https://huggingface.co/datasets/TongjiZhanglab/chrombert/resolve/main/hg38_6k_factors_list.txt)

## 在服务器上运行和验收

eQTL 和 STARR 的输入表可用 `benchmark.py extract-demo "$BENCHMARK_WORKSPACE/cache/demo.tar.gz" --destination "$BENCHMARK_WORKSPACE/cache/official-demo"` 提取。固定版本的 hg38 FASTA 和 `hg38.fa.fai` 放在资产的 `other/`，DNABERT-2 离线快照放在 `checkpoint/dnabert2-117m/`，官方调控因子总表及 UCSC `hg38ToHg19.over.chain.gz` 放在 `config/`。也可将任务表直接安装到服务器对应的 `demo/`。用 `benchmark.py audit --bundle ... --assets ...` 在 CPU 上核对原表、坐标、标签、划分及重叠。

STARR 原表为 hg19 坐标，ChromBERT 输入以 hg38 区间 ID 定位。审计用 UCSC 转换链将 hg38 区间中点映射到 hg19，并检查其位于对应教程区间内。沿用官方区间 ID。清除重叠后，正式划分为 26,208 条训练记录、4,082 条验证记录、721 个成对测试位点。

eQTL 和 STARR-seq 的划分审计保存到 `split_manifest.json`，包括原表规模、实际划分和重叠检查。STARR-seq 的推理吞吐同时记录位点对数和单状态前向数。

通过 `benchmark.py launch --workspace` 显式指定共享工作区。将示例中的 `2` 换成实际空闲的 GPU 编号。

```bash
python3 benchmarks/benchmark.py launch cistrome --mode preflight --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch cistrome --mode full --workspace "$BENCHMARK_WORKSPACE" --gpu 2
```

`cistrome-sampled` 读取工作区的 `assets/hg38_1kb/demo/prompt_imputation/MAZ_K562_sample250k_ids.csv`。配套 JSON 记录抽样种子、配额和校验值。可将全量运行的 `splits/test.csv` 传给 `benchmark.py sample-cistrome` 重建清单。

`preflight` 检查运行环境。默认的 `full` 模式按所选配置执行从数据准备到结果验收的流程，由 Docker 在后台运行。启动器检查 GPU 占用和本工作区的运行容器，并打印 `run.log` 与结果目录。成功时运行目录下的 `exit_code` 为 0，`result/status.json` 标为已验证，`result/check.json` 记录测试行匹配和重算分数。可以把运行目录或其 `result/` 子目录交给 `check`。

```bash
python3 benchmarks/benchmark.py check /absolute/path/to/official_run
```

如果服务器的 Python 缺少指标计算依赖，`check` 会借用实验室镜像，在 CPU 容器中复核。

对照内置 GELU 时，用 `--variant gelu-builtin --from-run <手写版运行目录>`。内置版从训练到推理都用 PyTorch 的 tanh GELU，并用本轮权重完成测试。Cistrome 沿用官方 prompt 权重做推理。`--gelu builtin` 是同一个变体的兼容入口。

端到端对照复用参考运行中经过哈希核对的划分。已有完整划分归档的指标验收失败运行也可提供划分。Profiler 复用参考权重时，参考运行须通过验收。两种模式都核对原始配置和资产，对照变体还要求相同的硬件及软件环境。

每次归档生效配置、官方 Notebook 快照、训练及测试划分、逐样本预测和微调权重。训练计时含模型初始化、验证及 checkpoint 保存。推理分别记录模型和数据加载器准备时间与逐批预测耗时，吞吐包含读数据和传输。显存记录 PyTorch 阶段峰值已分配及已预留字节。CPU 记录主进程与子进程 RSS 合计的 0.2 秒采样峰值。输入和生成文件分别记录字节数。

## 性能分析

| 模式 | 运行范围 |
| --- | --- |
| `profile` | 完成所选配置的端到端运行，同时采集局部 trace。GEP、EZH2 采集训练和推理，Cistrome、eQTL 采集推理。 |
| `profile-only` | 五项任务均支持。复用 `--from-run` 指定的已验收划分及推理权重。有训练的任务在第一次优化器更新后停止，推理执行至多 8 批，其中至多 4 批进入 trace。 |
| `smoke` | eQTL、STARR-seq 支持。每个划分取 16 条记录，训练 1 轮。 |
| `smoke-profile` | 在 eQTL 短跑中采集推理 trace。 |

以下命令从已验收的 STARR-seq 端到端运行采集 Profiler。

```bash
python3 benchmarks/benchmark.py launch starr --mode profile-only --from-run /path/to/verified-starr-run --workspace "$BENCHMARK_WORKSPACE" --gpu 2
```

结果目录保存 CPU/CUDA trace、算子用时和 `metrics.json` 中的 `profiling` 字段，实际采样批次以归档为准。比较端到端耗时、吞吐和显存时，使用 `full` 和 `"profile": false`。

## 维护和 CPU 回归测试

任务实现通过 `tasks/protocol.py` 的 `TaskProtocol` 注册。eQTL 在契约内继续使用原来的训练循环，STARR-seq 继续按位点保存两种状态及差值。运行器通过 `runtime.py` 预检和采样，通过 `predictions.py` 复用单样本预测，通过 `archives.py` 校验归档和对照参考。

`model_sources.py` 负责按来源打包、记录和校验模型材料，现有配置默认使用官方来源，打包时从 Git 中导出配置固定的官方提交。`variants.py` 注册实验变体，并负责声明参考运行要求和恢复实验开关。现有结果目录仍可用 `check` 复核。

装有 NumPy、pandas、SciPy 和 scikit-learn 的 Python 环境可以直接运行 CPU 测试。

```bash
PYTHONPATH=benchmarks python3 -B -m unittest discover -s benchmarks/tests -v
```
