# SurRoL-VLM 项目架构分析报告

## 1. SurRoL 环境背景知识
**SurRoL** 是一个开源的、以强化学习 (RL) 为中心的、兼容 dVRK (达芬奇研究套件) 的手术机器人学习平台。
- **核心特点**：它基于 PyBullet 物理引擎构建，提供了 Gym 风格的 API (兼容 Stable-Baselines3 等主流 RL 库)，内置了诸如 `StaticTrack` (静态跟踪)、`NeedlePick` (夹取缝合针)、`PegTransfer` (转移物块) 等 10 项手术相关的 RL 任务。
- **主要作用**：它不仅仅是一个仿真环境，还能够让研究人员通过强化学习来训练手术机器人的自动化动作策略。项目通过提供各类手术器械模型及评测代码，为闭环的具身智能算法（Embodied AI）提供实验床。

## 2. VLM 目录整体架构
[vlm](file:///e:/LLM/SurRoL-main/vlm/eval_reward_discrimination.py#16-33) 目录的作用是将 **视觉语言模型（Vision-Language Model, 特指 Qwen2-VL）** 融入到 SurRoL 的环境体系中。主要有两个方向的实践：
1. **VLA (Vision-Language-Action)**：将 VLM 直接微调为一个动作预测策略（即端对端模仿学习），输入当前图像与文本指令，模型直接输出机器人的连续动作（或离散动作参数）。
2. **RLAIF (Reinforcement Learning from AI Feedback)**：利用 VLM 作为 Reward Model（奖励函数），对当前的仿真状态图像进行评分（如 0-10 分），并将这个分数作为 PPO 强化学习算法的奖励信号，解决以往稀疏奖励（Sparse Reward）训练困难的问题。

以下为每个子目录与其对应的核心代码文件解析：

### 2.1 数据集生成与处理 (dataset/)
本目录主要用于从 SurRoL 内置的 Oracle（专家策略）中提取成功的数据，并处理为大模型 SFT（监督微调）格式。

- **[export_expert_universal.py](file:///e:/LLM/SurRoL-main/vlm/dataset/export_expert_universal.py)**：通用的专家数据生成流。针对任意指定的 SurRoL 任务（单臂 PSM 或内窥镜 ECM 任务）。调用环境内置的 [get_oracle_action()](file:///e:/LLM/SurRoL-main/vlm/eval/eval_closed_loop.py#125-131) 获取完美动作，每步保存观测状态、动作、RGB图像，并统一存入 `manifest.jsonl` 中。
- **[export_active_track_vla.py](file:///e:/LLM/SurRoL-main/vlm/dataset/export_active_track_vla.py)**：这是专门为 `ActiveTrack` 任务早期写的更简洁的数据导出脚本，作用类似。
- **[data_processing.py](file:///e:/LLM/SurRoL-main/vlm/dataset/data_processing.py)**：处理原始 `.jsonl` 数据的核心逻辑，提供获取动作极值（min/max归一化操作）、以及通过 K-Means 构建 VQ (Vector Quantization) 动作码本将连续动作离散化给不同架构 LLM 使用的功能模块。
- **[convert_manifests.py](file:///e:/LLM/SurRoL-main/vlm/dataset/convert_manifests.py)**：将导出的 `manifest.jsonl` 转变为 ShareGPT 格式（含多轮对话式 messages，如：用户提供图像与指令，AI回答下一步动作组合）的 `training_data.json`，方便接入诸如 LLaMA-Factory 等外部微调框架。
- **[vla_dataset.py](file:///e:/LLM/SurRoL-main/vlm/dataset/vla_dataset.py)**：供 Hugging Face `Trainer` 使用的 PyTorch [Dataset](file:///e:/LLM/SurRoL-main/vlm/dataset/vla_dataset.py#29-100) 和 [Collator](file:///e:/LLM/SurRoL-main/vlm/dataset/vla_dataset.py#102-209)，内部会自动将数据拼成大模型需要的文本聊天模板并读取相应的 PIL Image。另外 [VLACollator](file:///e:/LLM/SurRoL-main/vlm/dataset/vla_dataset.py#102-209) 还实现了 Loss 的掩膜（Label Masking），只对 Assistant 的回答（动作）计算 Loss。

### 2.2 评估分析模块 (eval/)
评价由于微调或强化学习得到的新策略的有效性。

- **[eval_closed_loop.py](file:///e:/LLM/SurRoL-main/vlm/eval/eval_closed_loop.py)**：对微调好的 VLA 模型在仿真环境中进行闭环测试：`截取环境画面图片 -> 传给 VLA -> VLA 输出动作 -> 环境 Step -> 循环`。支持本地推理与请求远程远端 Server 推理（为了前后端解耦）。
- **[eval_rlaif_zeroshot.py](file:///e:/LLM/SurRoL-main/vlm/eval/eval_rlaif_zeroshot.py)**：Zero-Shot 泛化测评，读取通过 RLAIF 训练出来的 SB3 (Stable-Baselines3) PPO 模型，在另一个从未见过的新任务场景上运行来检查迁移能力，还可以直接录制视频。
- **[plot_learning_curves.py](file:///e:/LLM/SurRoL-main/vlm/eval/plot_learning_curves.py)**：读取训练产生的 `experiment_metrics.csv`，将 Sparse Reward、人工稠密特征 Reward，以及 RLAIF VLM Reward 模式下的验证成功率曲线和回报曲线画图做消融实验对比。
- **[vla_server.py](file:///e:/LLM/SurRoL-main/vlm/eval/vla_server.py)**：一个轻量的 `http.server`。它不仅可以用于部署 VLA（根据传入图像实时给出动作预测）当作大脑服务端，从而避免在仿真环境容器里加载繁重的 Torch 和 Transformers。
- **(顶层文件) [eval_reward_discrimination.py](file:///e:/LLM/SurRoL-main/vlm/eval_reward_discrimination.py)**：一个有用的测试脚本。用 VLM 服务端去对一些“成功/失败”对比强烈的图片进行打分预测测试，判断它是否能拉开分差（区分度），进而证明当前的 Prompt 指令配置是否能够承担 Reward Model 的重任。

### 2.3 模型结构层 (model/)
- **[qwen_vl_vla.py](file:///e:/LLM/SurRoL-main/vlm/model/qwen_vl_vla.py)**：封装了对 Hugging Face `Qwen2VLForConditionalGeneration` 模型的读取和处理逻辑。提供了便捷的 [load_model_and_processor](file:///e:/LLM/SurRoL-main/vlm/model/qwen_vl_vla.py#35-97)（支持 BitsAndBytes QLoRA 量化及 Flash Attention），以及给模型挂载 PEFT 的 [apply_lora()](file:///e:/LLM/SurRoL-main/vlm/model/qwen_vl_vla.py#99-139) 注入模块和推理时的 LoRA 模型整合融合工具 [load_model_for_inference()](file:///e:/LLM/SurRoL-main/vlm/model/qwen_vl_vla.py#149-191)。

### 2.4 VLM 作奖励模型层的相关代码 (reward/)
- **[vlm_reward_scorer.py](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_scorer.py)**：定义了 [VLMRewardScorer](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_scorer.py#102-273) 类，负责根据 "有无目标参考图片 (With Goal/No Goal)" 构建精细刻薄的刁钻系统 Prompt 交给 VLM 打分，然后使用简单的正则化手段从 VLM 生成的高自由度语言文本中提取准确的数值（分数值），在 0-10 分之间进行归一化并缓存重用，以降低频繁调用的开销时间；还提供了调用远端 API 版本的 [RemoteVLMRewardScorer](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_scorer.py#276-370)。
- **[vlm_reward_wrapper.py](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_wrapper.py)**：符合 OpenAI Gym 规范的 Wrapper，将其套在 SurRoL 环境外面，用以劫持环境输出的原始奖励。提供了三种模式：完全替换原始稀疏奖励 (replace)、在原有基础上将 VLM 奖励进行加和 (add)、相乘 (multiply)。提供 `score_every` 间隔计分进一步削减少生成带来的推理耗时。

### 2.5 服务端 (server/)
- **[vlm_rpc_server.py](file:///e:/LLM/SurRoL-main/vlm/server/vlm_rpc_server.py)**：FastAPI 实现的服务端版本（包含 /score 端点），功能与 [vla_server.py](file:///e:/LLM/SurRoL-main/vlm/eval/vla_server.py) 相似，但它专门作为 Reward Server 进行 RLAIF 的评分，供 [RemoteVLMRewardScorer](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_scorer.py#276-370) 远程请求并返回得分统计信息。这种服务端架构可以方便将 VLM 放于带有大显存的服务器上独立运行，而不增加本地 RL 容器负载。

### 2.6 训练引擎 (trainer/)
- **[train_vla.py](file:///e:/LLM/SurRoL-main/vlm/trainer/train_vla.py)**：VLA 的监督微调主流程代码。拉起 YAML 配置文件设定的 Qwen2-VL 模型与 VLA Dataset，开启 LoRA/QLoRA 并使用 Hugging Face 的原版 `Trainer` API 进行梯度回传式的微调（端对端模仿学习）。
- **[train_rlaif.py](file:///e:/LLM/SurRoL-main/vlm/trainer/train_rlaif.py)**：RLAIF 训练的大脑。采用 Stable-Baselines3 的 PPO 算法进行 RL 训练。流程核心为：挂起 Oracle 获取完美的任务完成截图当作 Goal -> 初始化 [VLMRewardScorer](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_scorer.py#102-273) -> 把环境嵌套上 [VLMRewardWrapper](file:///e:/LLM/SurRoL-main/vlm/reward/vlm_reward_wrapper.py#31-151) -> SB3 PPO 根据 VLM 提供的高密度分值进行模型权重的学习；并在训练中通过 Callback 定期将效果进行记录（如调用失败率等）。
- **[infer_vla.py](file:///e:/LLM/SurRoL-main/vlm/trainer/infer_vla.py)**：命令行本地快捷推理单张图片或整个目录的小脚本，用于对微调后的 VLA 对静态帧的行为预测效果进行肉眼校准或功能调试。

## 总结
VLM 这条路线实现了：
1. **策略控制中心** (模仿学习路径)：从仿真中抽数据 -> 微调 Qwen VLM -> 部署模型当作独立服务接受图片输入输出控制动作信号 -> 回馈到仿真器。
2. **监督评估中心** (RLAIF 强化学习路径)：把 VLM 作为一个精准反馈分数的 Teacher -> 加载进 Gym 的 Reward Wrapper 中 -> 基于这个 Teacher 的分数指引，训练出一个很轻量的多层感知机或卷积 RL 机器人模型。

整体结构清晰，做到了模型推理环境与 RL 训练仿真环境的松偶合支持（不仅可通过本地直接注入代码，亦可通过 RPC 服务端/客户端通信剥离显存依赖）。
