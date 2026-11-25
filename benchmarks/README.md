# ChromBERT 官方任务 Benchmark

以 ChromBERT 官方教程为任务定义。每种任务在 `tasks/` 中有独立的 Python 文件。`training.py` 提供共用的 Lightning 训练逻辑，`runner.py` 组织运行、计时、导出和验收。所有操作从 `benchmark.py` 进入。原版模型从上游固定提交 `10589002597a9ef06613ed41b6ee83e8dd565c25` 打包，官方 Notebook 随结果保存。

## 任务

| 任务 | Benchmark 设置 | 输出 |
| --- | --- | --- |
| `gep` | [成纤维细胞 → 成肌细胞的表达变化](https://chrombert.readthedocs.io/en/latest/tutorial_transdifferentiation_transcriptome.html)。用官方两份 TPM 表生成 `log1p(TPM)` 差值。seed 55 抽取 80/20/50 个训练、验证和测试样本。保留 9 窗口、2 轮预算和按验证 RMSE 选权重的规则。 | 用本轮权重和官方全量微调权重分别预测同一批 50 个基因，报告 Pearson、Spearman、MSE、MAE、R²。 |
| `gep-all` | 同一任务和 seed 55 划分，使用全部 15,720/1,965/1,965 个样本。训练 2 轮，每轮结束评估完整验证集，按验证 RMSE 选权重。 | 用本轮权重预测完整测试集并报告同样的五项指标。 |
| `ezh2` | [人胚胎干细胞 EZH2 位点分类](https://chrombert.readthedocs.io/en/latest/tutorial_locus_specific_TRN_ezh2.html)。使用官方峰文件和 seed 42 的完整 8,806/1,101/1,101 划分，排除 H3K27me3 输入轨道。训练 1 轮，每 128 个训练批遍历完整验证集。 | 用末轮权重评估完整测试集，记录 AUROC、AUPRC、BCE。末轮权重和验证 BCE 最优权重分别保存。 |
| `cistrome` | [MAZ × K562 的在线 DNase prompt 补全](https://chrombert.readthedocs.io/en/latest/tutorial_prompt_cistrome_imputation.html)。加载官方 prompt 权重，对 hg38 1 kb 参考表的全部 2,137,894 个区间推理，保留原始类别比例。 | 保存逐区间 logit 和概率，在完整测试集上计算 AUROC、AUPRC、BCE。 |
| `cistrome-sampled` | 同一模型、权重和 prompt，使用固定的 250,000 个区间。按染色体和标签分层，保留 7,439 个正样本和 242,561 个负样本，覆盖 24 条染色体。 | 用固定区间报告 AUROC、AUPRC、BCE，以及推理时间和吞吐。 |
| `eqtl` | 肺组织变异分类。官方 `prompt_dna` 将同一区间的 ChromBERT 表征与 DNABERT-2 编码的替代序列融合。将 1,535 条记录按 1 kb 区间分组，约 60%/20%/20% 划分。 | 验证 BCE 选权重，导出变异 ID、logit 和概率，报告 AUROC、AP 和 BCE。 |
| `eqtl-official` | 同一肺组织数据，复现教程 seed 42 随机划分的 921/614 条记录和训练循环。 | 用末次权重报告同样的分类指标。单列结果和区间重叠。 |
| `starr` | 保留官方 721 个成对评估位点。以官方训练、验证表为起点，剔除与成对测试重叠的区间。保留官方非 TF 排除和轨道扰动实现。 | 保存野生型、扰动组及差值的实测与预测结果，报告三个 Pearson、差值 RMSE 和预测方差。 |

以上任务沿用官方教程的预测流程。各配置的评价范围见上表。论文实验及教程中的缓存、RNA 和单细胞 prompt 示例见原资料。[论文与补表](https://pmc.ncbi.nlm.nih.gov/articles/PMC13069865/#supplementary-material)

## 可解释性

`ezh2` 和 `gep-all` 使用本轮测试预测所用的 checkpoint 提取调控因子嵌入。EZH2 在经典和非经典位点组内分别平均嵌入，按候选因子与 EZH2 的余弦相似度及两组差值排序。全量 GEP 使用留出的测试基因，按上调和表达近似不变分组，以两组平均嵌入的余弦相似度从低到高排序。

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

eQTL 和 STARR 支持 `smoke`，每个实际划分取 16 条记录并训练一轮。eQTL 支持 `smoke-profile`，单独标记两个编码器和融合头。全量输出含 `split_manifest.json`，保存原始行数、实际划分和重叠。STARR 的推理吞吐同时记录位点对数和单状态前向数。

通过 `benchmark.py launch --workspace` 显式指定共享工作区。将示例中的 `2` 换成实际空闲的 GPU 编号。

```bash
python3 benchmarks/benchmark.py launch gep --mode preflight --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch gep-all --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch ezh2 --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch cistrome-sampled --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch eqtl --mode smoke-profile --workspace "$BENCHMARK_WORKSPACE" --gpu 2
```

`cistrome-sampled` 的固定清单位于工作区的 `assets/hg38_1kb/demo/prompt_imputation/MAZ_K562_sample250k_ids.csv`。同名 JSON 记录生成时使用的 seed 42、各染色体配额和 SHA-256。运行时按清单取区间并核对标签；种子用于追溯生成过程。配额保留了 chrY 唯一的正样本，chr1 的正样本配额相应减一。若要复建清单，可用 `benchmark.py sample-cistrome <全量 splits/test.csv> --destination <空的输出目录>`。

`preflight` 检查运行环境。默认的 `full` 模式按所选配置执行从数据准备到结果验收的流程，由 Docker 在后台运行。启动器检查 GPU 占用和本工作区的运行容器，并打印 `run.log` 与结果目录。成功时运行目录下的 `exit_code` 为 0，`result/status.json` 标为已验证，`result/check.json` 记录测试行匹配和重算分数。可以把运行目录或其 `result/` 子目录交给 `check`。

```bash
python3 benchmarks/benchmark.py check /absolute/path/to/official_run
```

如果服务器的 Python 缺少指标计算依赖，`check` 会借用实验室镜像，在 CPU 容器中复核。

对照内置 GELU 时，用 `--variant gelu-builtin --from-run <手写版运行目录>`。内置版从训练到推理都用 PyTorch 的 tanh GELU，并用本轮权重完成测试。Cistrome 沿用官方 prompt 权重做推理。`--gelu builtin` 是同一个变体的兼容入口。

端到端对照复用参考运行中经过哈希核对的划分。已有完整划分归档的指标验收失败运行也可提供划分。Profiler 复用参考权重时，参考运行须通过验收。两种模式都核对原始配置和资产，对照变体还要求相同的硬件及软件环境。

每次归档生效配置、官方 Notebook 快照、训练及测试划分、逐样本预测和微调权重。训练计时含模型初始化、验证及 checkpoint 保存。推理分别记录模型和数据加载器准备时间与逐批预测耗时，吞吐包含读数据和传输。显存记录 PyTorch 阶段峰值已分配及已预留字节。CPU 记录主进程与子进程 RSS 合计的 0.2 秒采样峰值。输入和生成文件分别记录字节数。

## 性能分析

五项任务共用 `profiling.py`。将 `configs/<任务>.json` 中的 `"profile"` 设为 `true`，用原有 `full` 命令运行即可开启。也可在命令行临时开启。

```bash
python3 benchmarks/benchmark.py launch gep --mode profile --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch ezh2 --mode profile --workspace "$BENCHMARK_WORKSPACE" --gpu 2
python3 benchmarks/benchmark.py launch cistrome --mode profile --workspace "$BENCHMARK_WORKSPACE" --gpu 2
```

已有正式运行结果时，可以只采集 Profiler。以下命令复用已验收的划分和推理权重。训练阶段走到第一次优化器更新，推理阶段最多走 8 批。结果目录保存采样范围、trace、算子报告及验收记录。

```bash
python3 benchmarks/benchmark.py launch starr --mode profile-only --from-run /path/to/verified-starr-run --workspace "$BENCHMARK_WORKSPACE" --gpu 2
```

GEP 和 EZH2 采集首轮训练和本轮权重的测试推理。Cistrome 采集测试推理。训练采集在一次优化器更新时结束，推理最多采集 4 批。结果目录保存 CPU/CUDA 时间线、算子报告和 `metrics.json` 中的 `profiling` 字段。报告包括编码器、Attention、FFN、GELU 和预测头，另外按模型标记 GEP 的窗口池化与 Cistrome 的 prompt 池化。正式比较整体耗时、吞吐和显存时，将 `"profile"` 设为 `false` 并使用 `full`。

## 维护和 CPU 回归测试

任务实现通过 `tasks/protocol.py` 的 `TaskProtocol` 注册。eQTL 在契约内继续使用原来的训练循环，STARR-seq 继续按位点保存两种状态及差值。运行器通过 `runtime.py` 预检和采样，通过 `predictions.py` 复用单样本预测，通过 `archives.py` 校验归档和对照参考。

`model_sources.py` 负责按来源打包、记录和校验模型材料，现有配置默认使用官方来源，打包时从 Git 中导出配置固定的官方提交。`variants.py` 注册实验变体，并负责声明参考运行要求和恢复实验开关。现有结果目录仍可用 `check` 复核。

装有 NumPy、pandas、SciPy 和 scikit-learn 的 Python 环境可以直接运行 CPU 测试。

```bash
PYTHONPATH=benchmarks python3 -B -m unittest discover -s benchmarks/tests -v
```
