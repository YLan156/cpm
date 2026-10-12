# 情绪×交流目的：HPC 实验脚本包

此包包含真实已有LoCoMo记忆的子集、20个AI构造情境和固定候选，不需要原项目其余目录。含196条唯一记忆，实际每情境取6条候选、返回3条。不是新人工金标准，不包含模型权重、密码或API密钥。

## 三组比较

- A_goal：当前问题＋交流目的。
- B_joint：再提供当前情绪。
- C_mechanism：信息与B相同，增加按用途选择的明确策略。

三组看到相同原文、相同暂定抽取字段及候选顺序。事实问题在所有组均保留完整事实请求。字段是事件、情绪、实际行动、结果，只允许原文引用或null；不符合原文的引用转为unknown，原始输出仍保留。引用一致不能保证理解正确。输出非法时记失败，不用基线替代。

C是提示词机制，不是训练出的新网络。情绪和目的由条件提供，未自动识别。当前没有帮助性金标准，不会自动生成“方法有效”的结论。

## 1. 检查GPU（在你申请的GPU开发环境或作业内）

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

不要在集群登录节点运行模型。如果平台已提供八卡开发容器，可直接在容器终端执行后续命令，不需要重复提交Slurm任务。

## 2. 安装环境

建议Python 3.10—3.12。若平台已有CUDA版PyTorch，使用：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/check_inputs.py
```

若没有PyTorch，按集群提供的CUDA环境或PyTorch官网安装匹配版本。举例：驱动支持CUDA 12.4时可使用下面命令；这不是所有HPC通用的安装命令：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 3. 准备模型

先根据显存选择模型。默认示例Qwen2.5-7B-Instruct，半精度权重约14GB，另需运行显存；24GB以上单卡较适合作为起点，不能保证所有输入都不溢出。显存不足可以先用3B做流程检查，不能把它称为强模型验证。

```bash
python src/download_model.py --repo Qwen/Qwen2.5-7B-Instruct --output models/instruct
```

下载器将main解析为固定提交、下载权重并保存SHA256及provenance.json；分片safetensors模型也支持。下载需要能访问Hugging Face；集群若不能联网，可在有网络机器运行相同下载器，将整个模型目录上传。现有模型也可通过CPM_LLM_MODEL_DIR指定，但须包含下载器生成的provenance.json。不要手工伪造版本号或校验记录。

## 4. 先单卡试跑

```bash
bash scripts/run_single.sh
```

先执行两条合成抽取检查（原文引用、已发生行动与未来计划），失败即停止。通过也只代表这两条检查，不保证总体语义正确。然后运行1个开发情境＋2个不同对话的探索测试情境，共54份排序。A在两种情绪条件的提示相同，会复用缓存；这不是额外独立样本。

结果在results/pilot/。观察smoke.json、annotations.jsonl和model_calls.jsonl；程序不会偷偷把失败抽取换成关键词规则。

## 5. 八卡并行跑完整20情境

确认模型和抽取输出可用之后，在已分配八卡的环境执行：

```bash
bash scripts/run_multi.sh
```

总共20×2情绪×3目的×3方法=360份排序。八张卡各自加载一份模型、处理不同情境，不是八卡合并显存。单卡必须能容纳该模型。已有实验对话划分保留，没有训练、调参或新确认测试集。

卡数可减少：

```bash
CPM_SHARDS=2 bash scripts/run_multi.sh
```

使用调度器分配的CUDA_VISIBLE_DEVICES，不绕过资源配额。不要同时启动多个运行占用同一GPU。若使用Slurm，将scripts/slurm_example.sh中的资源设置按学校说明调整，再从包根目录提交：

```bash
sbatch --partition=实际分区 --account=实际项目 scripts/slurm_example.sh
```

本包没有自动提交任何超算作业。分区、项目、时限及GPU资源语法因集群而异。

## 6. 看结果

- summary.csv：有效输出数；构造证据是否在池、是否进入前三（只在fact条件解释）。
- contrasts_summary.csv：B相对A、C相对B的排序集合变化；不是效果提升。
- blind_comparisons.json：未评分的匿名左右比较，包含相同输出。
- review_key.json：单独的方法映射，不要提供给盲评者。
- report.md：执行摘要与解释限制。
- 每个分片的model_calls.jsonl、annotations.jsonl、rankings.jsonl：完整输入、原始输出和选择结果。

结果目录使用相同配置可续跑缓存。换模型、精度或实验配置请用新的CPM_OUTPUT_DIR，例如：

```bash
CPM_LLM_MODEL_DIR=/path/to/other/model CPM_OUTPUT_DIR="$PWD/results/other-model" bash scripts/run_multi.sh
```

汇总器禁止混合模型或重复条件。少量情境和它们的多次变体不是大量独立样本。没有生成最终回答，不能报告回答准确率、心理疗效或独立人工金标准。

## 验证状态

交付前已实际检查数据校验、人称/时间约束、对话划分，Python与shell语法，以及使用明确标记的合成输出检查结果汇总流程。当前编写环境无GPU，未运行本包7B模型或CUDA分支，不声称八卡运行已通过。之前1.5B CPU量化调用失败，未作为本包成功结果；这些脚本不会启用那条量化路径。

拿到结果后，将results/目录（无需模型权重）打包返回即可分析。先看原文和模型输出，不必立刻扩大人工标注。
