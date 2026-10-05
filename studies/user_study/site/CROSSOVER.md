# 12 人 × 6 题交叉研究方案

每人每道正式题目只做一次，其中三题使用 Zinnia、三题使用 Rust/RISC Zero。共 72 次正式任务观测，独立参与者数量仍为 12。

## 分配与随机化

采用六阶 Williams 顺序，每条配两条互补语言序列。创建 cohort 时，将 12 个模板随机映射到参与者链接，并保存实际分配。按预先规定的报名顺序发放链接，不根据能力或已观察成绩挑选模板。

- 每题每种环境六人；每题在每个位置各出现两次。
- 每个位置的两种环境各六人；每对不同题目的有向相邻关系各出现两次。
- 语言序列包括连续使用和切换，避免当前语言与前一语言完全绑定。
- 首次练习环境与首题正式环境的四种组合各三人。

编号表示当前任务集的固定任务身份，位置表示完成顺序。Z=Zinnia，R=Rust/RISC Zero。

| 模板 | 位置1 | 位置2 | 位置3 | 位置4 | 位置5 | 位置6 | 练习顺序 |
|---|---|---|---|---|---|---|---|
| S01 | 1Z | 2Z | 6R | 3Z | 5R | 4R | Z → R |
| S02 | 1R | 2R | 6Z | 3R | 5Z | 4Z | R → Z |
| S03 | 2Z | 3R | 1Z | 4R | 6R | 5Z | R → Z |
| S04 | 2R | 3Z | 1R | 4Z | 6Z | 5R | Z → R |
| S05 | 3R | 4Z | 2R | 5R | 1Z | 6Z | R → Z |
| S06 | 3Z | 4R | 2Z | 5Z | 1R | 6R | Z → R |
| S07 | 4Z | 5R | 3R | 6Z | 2Z | 1R | R → Z |
| S08 | 4R | 5Z | 3Z | 6R | 2R | 1Z | Z → R |
| S09 | 5R | 6R | 4Z | 1Z | 3R | 2Z | R → Z |
| S10 | 5Z | 6Z | 4R | 1R | 3Z | 2R | Z → R |
| S11 | 6R | 1Z | 5Z | 2R | 4Z | 3R | Z → R |
| S12 | 6Z | 1R | 5R | 2Z | 4R | 3Z | R → Z |

## 流程与记录

知情说明 → 背景问卷 → 第一种环境教程及不限时练习 → 第二种环境教程及不限时练习 → 六道正式任务。

两次练习都有独立草稿、运行记录和通过门槛，不计时、不隐藏评分、不计入正式题数。正式题前说明该题环境。参与者可切换界面语言，但不能更改实验分配。每题开始前可以休息；暂停、关闭页面、提交评分期间继续按原规则停表。

六道正式任务每题均为 10 分钟，时间上限合计 60 分钟，另加双环境教程、练习和休息。任务说明、公开和隐藏测试、预算及文档支持对两种环境一致。

执行环境在每道题开始时冻结到 session，并写入每次公开运行、隐藏评分和最终提交；后台重放和重试沿用该份代码的环境。完整 attempt trail 保留代码、版本、有效时间、运行耗时及隐藏成绩，并增加 condition 与 period 字段。两个练习使用 practice_zinnia 和 practice_rust 的独立存储身份。

## 导出与历史数据

导出 schema 3；研究者可按 cohort 单独导出 JSON 和参与者链接 CSV。

- participants.design 为 crossover-12x6-v1。
- task_order、task_conditions、practice_order、schedule_slot 保存实际分配。
- 对交叉参与者，旧字段 participants.condition 只表示首个练习环境，用于兼容；**正式分析使用 sessions/runs/attempt_trail 中的 condition 与 period**。
- cohort ID 隔离本批 12 人与以前的 pilot/其他设计。预览链接不进入研究导出。
- 既有链接保持原来的任务版本、固定语言、预算、代码与成绩。

顺序平衡不能保证消除不对称的学习效应。分析应考虑参与者、任务和位置；同一人的多题、多次尝试不能视为独立参与者。正式收集前仍须固定主指标、超时处理和排除规则。失访数据保留，不按观察到的结果重新分配条件。

## 验证

单元测试检查全部模板的参与者、任务、位置、相邻题目和练习顺序平衡，以及双练习门槛、语言冻结、隐藏重放、重试和 cohort 隔离。

交叉浏览器测试创建完整 12 人 cohort，完成一对互补模板的两名测试参与者：4 次练习、12 次正式运行、12 次最终评分和 12 次隐藏尝试评分，均经过实际 Zinnia/RISC Zero 环境。旧版单环境浏览器流程也通过；站点回归测试共 36 项通过。

复现：npm run test:crossover --prefix studies/user_study/site；分配测试：.venv/bin/python -m unittest studies.user_study.site.tests.test_crossover。测试均使用临时数据库，不写入正式研究记录。

### Matrix practice (v2)

Before the six timed tasks, everyone completes the same untimed, unscored
matrix exercise once in Zinnia and once in Rust on RISC Zero. Subtract a
per-column baseline from a matrix, clamp each negative difference to zero,
and sum across columns to return one result per row. The examples include
rectangular matrices, negative baselines, single rows/columns, and all-zero
outputs. All four examples are public; both practice gates require passing
with the current code. Hints, bilingual descriptions, and searchable manuals
are available. Array operations and pure loops are equally accepted.

The existing counterbalanced practice order is retained. Using the same
exercise in both environments controls algorithm difficulty during training;
these results are not scored as formal-task outcomes. New assignments store
practice_version=2 and use practice_zinnia_v2 / practice_rust_v2. Historical
addition contracts and their saved runs remain available under the old IDs.

### Moderate difficulty (revision 5)

New assignments use the six tasks in `../revision5/README.md`, at 600 seconds
each. Task order and language allocation remain balanced. All six have new
IDs; historical assignments retain their original contracts.
