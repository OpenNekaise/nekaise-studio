# nekaise-studio 重构指令 (REFACTOR.md)

> 目的:把 studio 从"当前可用"重构到"目标架构"。每一项写明动机 (why)、改造内容 (what)、验收标准 (done when)。
> 执行者注意:本文档基于架构描述而非逐行代码审查,遇到与现状冲突处,以本文档的原则为准、以最小改动路径实现,并在 PR 描述中记录偏差。
> (收到于 2026-07-21;本仓库 nekaise-gym 即为本指令的执行产物;偏差记录在 docs/REFACTOR-NOTES.md。)

---

## 第一优先级(先做,影响所有后续工作)

### R1. 抽出 gym 边界:任务 + 验证器 + runner 与训练解耦
**Why**: 验证器是全项目唯一的"对错定义",目前散落在 studio 内部 (packs/、judge、nekaise_bench 调用)。训练奖励、SFT 过滤、评测必须用同一份定义,且不绑定训练框架。
**What**:
- 新建顶层包 `gym/`(暂不拆独立 repo),包含 `gym/tasks/`、`gym/verifiers/`、`gym/runner/`。
- 验证器全部重构为纯函数:`verify(prompt: str, response: str, meta: dict) -> float`,无框架依赖、无全局状态。TRL 的 reward 包装器、SFT 过滤器、评测 runner 都只做薄适配。
- `gym/` 禁止 import studio 的任何模块(单向依赖:studio → gym)。
- 任务三件套入库标准:题面 + 参考解 + 验证器绑定,缺一不收;入库时跑"金标解必须通过验证器"的冒烟测试。
- 注意术语:gym 内的运行器命名为 `runner/`,"harness" 一词保留给 agent 运行时 (opennekaise/edge),避免一词两义。
**Done when**: `grep -r "from studio" gym/` 为空;同一个验证器函数被 reward、过滤、评测三处 import。

### R2. 写入 SPEC.md(宪法),放入 immutable bootloader 层
**Why**: 开发与实验由 agent 执行,agent 的行为由可读文件决定;不落盘的共识会被后续 agent 实例重新发明或违反。
**What**: SPEC.md 包含以下四节,`run-experiment` 技能的裁决逻辑必须显式引用:
1. **算法卡**(见 R3 表格)。
2. **禁用清单**: 课程学习、动态采样/难度过滤、长度惩罚(唯一例外:超长回复占比实测超标时,按 JustRL 两阶段长度调度加入)、PPO+critic、过程奖励模型、MCTS/搜索增强、DPO 叠加、模型融合。任何清单内技巧入库需人类批准。
3. **零假设元规则**: 任何算法改动声称有效前,必须击败对照组"同一基线多训等量算力"。两者同涨则判定改动无效。
4. **环境锁**: Unsloth/TRL/vLLM 版本三元组 pin 死 (uv lock);agent 无权变更依赖版本,环境变更是人类批准操作。
**Done when**: SPEC.md 进 bootloader;run-experiment 的裁决 prompt 中包含对 SPEC 的引用与零假设检查项。

### R3. 训练循环标准化:全部收敛到 Unsloth/TRL 原生 Trainer
**Why**: 自研训练循环是维护负担与 bug 来源;TRL 覆盖全部所需算法,包括 OPD (GKDTrainer)。
**What**: 每阶段一个冻结的 config 文件,agent 可动的旋钮仅限"数据配方"列:

| 阶段 | 实现 | agent 可动 | 冻结 |
|---|---|---|---|
| CPT | Unsloth 标准训练 + WSD 调度 | 数据配比 (domain/replay)、annealing 子集 | 优化器、调度形状、序列长度 |
| SFT | TRL SFTTrainer | 合成策略、judge 过滤阈值、混合比 | 损失函数 |
| RLVR | TRL GRPOTrainer + gym 验证器,JustRL 超参照抄 | 任务集组成 | 全部超参、clip-higher 为唯一稳定化手段 |
| OPD | TRL GKDTrainer,sampled-token 变体 | 教师选择 (RL 后自家检查点 / 特权信息自蒸馏) | 散度设置,不用 top-k 花样 |
| Agentic | 轨迹 rejection SFT → 短程 GRPO (outcome 奖励) | 任务集组成 | 5–15 步上限,禁止长轨迹稠密监督 |

- JustRL → TRL 的超参逐项映射表写进 SPEC.md 附录(clip 范围、KL 系数、rollout 数、温度),防默认值偏差。
- 自研循环代码移入 `attic/` 隔离,不删除但不再被引用。
**Done when**: 每阶段入口是 `python -m studio.stages.<stage> --config configs/<stage>.yaml`;configs/ 目录下无 agent 写权限之外的散落超参。

### R4. vLLM 接线
**Why**: 采样占 RL 时间 70–80%,rollout 引擎是吞吐第一杠杆。
**What**:
- GRPO rollout: Unsloth `fast_inference=True`(vLLM 共享权重模式),或 TRL colocate。禁止 transformers `generate()` 出现在任何 rollout/评测/数据生产路径。
- gym runner 与数据工厂:独立 `vllm serve` 进程,OpenAI 兼容接口;gym 考本地模型与考 API 模型走同一条代码路径。
- OPD 教师打分暂用 GKDTrainer 默认 (HF 前向),roadmap 记 `prompt_logprobs` 加速项。
- CPT/SFT 不引入 vLLM。
**Done when**: `grep -rn "\.generate(" --include="*.py"` 在 rollout/eval/datagen 路径下为空。

---

## 第二优先级(仪表盘可信化,先于任何新实验)

### R5. 可复现性冒烟测试与噪声带
**Why**: 小语料 + RL 的 seed 方差可能淹没实验效应;keep/rollback 对噪声做决策会让 LOG.md 积累假结论。
**What**:
- 新增工具 `studio.tools.variance_check`:同一配置 2–3 seeds,输出各指标的噪声带 (std)。
- 裁决规则改为:效应量 > 噪声带才允许 keep;噪声带数值写入 LOG.md 每条记录。
- 首个任务:对当前主力配置跑一次,校准历史结论,受影响的 crystallized skill 打 `unverified` 标。
**Done when**: LOG.md 新记录含 noise_band 字段;run-experiment 裁决逻辑读取它。

### R6. 难度阶梯与标定作业
**Why**: benchmark 的价值在区分度不在难度;全难题对 1B 学生无量程,对 GRPO 无梯度 (全错组 advantage=0)。
**What**:
- 新增 `gym/tools/calibrate.py`:用模型梯队 (API 强模型 / 7B / 1.7B / 目标底座,各采样 4 次) 跑全部任务,记录逐任务通过率,写入任务元数据 `difficulty_band`。
- 训练采样器按当前学生通过率过滤 20–80% 区间任务;<5% 入 `frontier` split(留天花板),>95% 入 `graduated` split(回归测试防遗忘)。
- eval split 冻结 + git 版本化;difficulty 标定定期重跑(学生变强后量程漂移)。
- 缺档任务由 API 参照难题合成简化变体补齐。
**Done when**: 每个任务有实测 difficulty_band;训练数据加载器带通过率过滤;eval/frontier/graduated 三个 split 物理分离。

### R7. anchor-recall 防 hack 与验证器硬化
**Why**: anchor-recall 可被"堆砌 anchor 字符串"刷分;judge 打分可被讨好。奖励防御靠测量,不靠给算法加补丁。
**What**:
- 监控项:高奖励样本的 anchor 密度分布、回答"清单化"程度 (格式特征),异常上升触发人工 review。
- 逐步扩充硬可验证任务类型:可执行 SPARQL (结果集 F1)、数值容差比较、沙箱单元测试;目标是程序化验证器奖励占比随版本上升,judge 退居不可验证领域兜底。
- 沙箱要求:超时、内存上限、无网络、测试用例不落盘。
**Done when**: 仪表盘含 anchor 密度曲线;gym/verifiers/ 下 SPARQL 与数值验证器可用并接入任一训练任务。

---

## 第三优先级(循环强健化)

### R8. LOG.md schema 化
**What**: 每条记录固定字段:hypothesis / variable / expectation / result / noise_band / verdict / confidence;同步写一份 JSONL (机器可读),LOG.md 保持人读版。历史记录不回填,新记录起强制。
**Done when**: run-experiment 输出两种格式;JSONL 可被脚本聚合出"假设命中率"统计。

### R9. 止损阀三件套
**What**:
- checkpoint 保留策略:每实验只留 best + last,其余训后自动清理;
- 单实验 wall-clock 与算力预算硬上限,超时强制中止并记 LOG;
- crystallize-skill 准入:结论须跨 ≥2 个独立实验复现;prune-skills 定期用零假设重审已结晶技能,不过者降级为 hypothesis。
**Done when**: 三条均为代码强制,而非技能 prompt 里的建议。

### R10. 隐私与 holdout 改为代码强制
**Why**: 现为约定 (环境变量 + 纪律),约定会被 agent 遗忘。
**What**:
- 库入口处断言 `NEKAISE_HOLDOUT` 已设置,未设置直接 fail-fast 并给出设置指引;
- git pre-commit hook:扫描合作方楼宇真实名称词表 (词表本身放在 git 之外的本地路径),命中即拒绝提交。
**Done when**: 未设 holdout 时任何楼宇相关入口无法运行;含敏感名的提交被 hook 拦截。

---

## 第四优先级(新管线段立项,可以只搭骨架)

### R11. agentic 训练段脚手架
**Why**: 当前循环优化"答题",产品需要"干活";此缺口是方向性的,先立项防止长期停留在叙事层。
**What**:
- `studio/stages/agentic/`:轨迹收集 (rejection SFT) 与短程 GRPO 两个入口,配置遵循 R3 表格;
- 跨仓库依赖:向 opennekaise 提 issue,要求 headless 可编程模式 (一条命令跑一个任务、结构化输出);该接口同时服务 gym 考试、RL rollout、CI 回归三处;
- gym 增加 agentic 任务类型:5–15 步、outcome 验证器 (产物比对 / MAE / 测试通过)。
**Done when**: 骨架代码 + 接口定义合入;端到端跑通 1 个玩具任务 (可以不训练,只验证轨迹采集→验证器→分数链路)。

### R12. 学生尺寸决策落地
**What**: 主线切换为 1B 底座 (配置默认值),3B 保留为可选验证线;产物命名遵循 `Nekaise-1B(-Base/-SFT)` 体系。若保留 3B 主线需在 SPEC.md 记录理由与"3B 结论何时需在 1B 复验"的规则 (learnability gap 风险)。
**Done when**: configs 默认指向 1B;README/STATUS 的目标描述与命名一致。

---

## 明确不做 (与做什么同等重要)
- 不迁移 veRL/OpenRLHF。触发器 (写入 SPEC.md):多卡多机 / 多轮 agentic RL 需在线 harness rollout / vLLM colocate 开满后吞吐仍是硬瓶颈。触发后也可只迁 agentic 段。
- 不引入禁用清单中的任何技巧。
- 不在当前语料规模 (3.5M token) 上做超参搜索类实验——结论不外推;agent 的自由度导向 corpus 产能与任务集扩建。

## 建议执行顺序
R2 (SPEC) → R5 (方差校准) → R1 (gym 抽取) → R3+R4 (循环标准化) → R6+R7 (阶梯与防 hack) → R8–R10 → R11+R12。
R2 与 R5 合计约一两天,但决定其后所有工作的可信度,严格先行。
