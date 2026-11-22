# Obsidian 写作材料审阅入口

此入口复用 AI-Scientist-v2 的 `perform_review`，替换默认会议审稿表。
输入是 AI-Researcher 输出的 Markdown 和可选证据摘录，不运行实验或生成论文 PDF。
第一阶段仅产生单份审阅，不做多审阅人共识，也不自动修改原稿。

## 安装及模型

在项目目录用 PowerShell 执行。审阅环境独立于实验环境。

```powershell
uv --no-cache venv .venv-review --python 3.11
uv --no-cache pip install --python .venv-review/Scripts/python.exe -r requirements-review.txt
```

模型服务使用本项目已有 `.env`。默认模型为 `qwen3.8-flash`，可用
`--model` 或 `AI_SCIENTIST_REVIEW_MODEL` 替换为本地适配层支持的模型。
密钥不会写入结果文件。

## 本次复现命令

```powershell
.\.venv-review\Scripts\python.exe review_obsidian.py `
  --direct `
  --input 'E:\oblidian\file\safetyAI\Agent-IFC\30_Projects\AI-Researcher\Results\ai_researcher_ideation_20261023_213433_871020.md' `
  --evidence 'E:\xidian-ai-security-class-project-new-main\projects\project-01-ai-researcher\logs\log_2026-10-23 21-30-53.log' `
  --output-dir 'E:\oblidian\file\safetyAI\Agent-IFC\30_Projects\AI-Scientist-v2\Reviews'
```

输入日志时只提取唯一的 `OBSIDIAN_EVIDENCE_BUNDLE` 内容，不发送其他日志。
本次本地默认代理端口拒绝连接，因此命令加了 `--direct`，仅本次 API 调用忽略环境代理。
需要代理访问模型服务的环境请去掉此选项，不修改全局代理设置。
也可直接传入 UTF-8 证据 Markdown。省略证据时仅检查文稿内部一致性和表达。
加 `--dry-run` 只保存输入与指令，不导入模型库、不调用 API。
超过 45,000 字符会明确报错，需主动缩小材料，不会静默截断。

## 输出及后续操作

每次运行创建独立目录，包括 `input.md`、`instructions.txt`、`manifest.json`。
成功时增加 `review.md`、`review.json`、`model_response.json`。
`manifest.json` 的 `status` 表示运行状态：`prepared` 为仅准备输入，`running` 为请求中，
`failed` 为执行或结构校验失败，`needs_attention` 为已保存部分意见但存在无法定位的引文，
`completed` 为输出校验完成。以上均不代表人工审核通过，`author_status` 仍为 `pending_author_review`。
独立入口给每个请求设置 180 秒超时，禁用旧函数的无限重试与 SDK 自动重试。
不修改实验入口的默认重试策略。
默认输出上限为 2,400 tokens，以减少长审阅的等待。可用 `--max-output-tokens` 调整；
超限导致 JSON 不完整时会报失败，保留模型输出，不会把截断结果当成有效审阅。
失败时保留输入与失败类型，不把不完整结果标为通过。

文稿哈希与证据哈希用于追踪版本。审阅问题必须包含文稿逐字短引文；
结构校验仍会拒绝字段不全、重复编号及“阻断”的 Minor 项。
引文不匹配时，只自动恢复唯一匹配的成对 Markdown 加粗或行内代码标记差异，
不删除空白、标点或使用模糊语义匹配。其他意见完整保存到 `unresolved_issues.json`，
并在 `review.md` 的待处理清单显示；有效意见仍可读取，运行状态为 `needs_attention`。
此时程序退出码为 0 仅表示结果已保存，调用方须读取 manifest 状态，不能据此认为审核通过。
如需沿用遇到引文错误即失败的行为，可加 `--strict-quotes`。
这种检查只确认文稿引文存在，不证明证据位置正确或语义判断成立。
原始论文 PDF 未重新核验，模型审阅仍需用户确认。

下一阶段可把确认后的 `review.md` 与原稿一起交给 AI-Researcher 修订，
修订稿另存。当前入口没有自动修订、自动批准或界面按钮。

## 复用已保存的模型结果

若模型已返回、但解析或校验失败，可在原命令上增加 `--resume-dir '失败运行目录'`。
入口要求输入快照哈希与原运行完全一致，读取原始模型响应，不再次调用模型。
已完成的目录不能覆盖。`needs_attention` 目录允许继续恢复。
`model_response.json` 保留收到的模型历史，恢复过程不覆盖它。
`review_received.json` 保存本次处理收到的结构化对象，`review.json` 为可定位的意见；
自动与人工校正、待处理清单均写入 manifest。整体判断及修订顺序未自动重写，
若有待处理意见，也不能把这些总括建议当作已校验结论。

例如，在原有命令上增加以下参数（沿用相同的 input、evidence 和 output-dir）：

```powershell
--resume-dir '原运行目录'
```

若仅有 Markdown 标记差异，可以直接离线恢复。其他引文须人工定位并提供下面的显式校正；
不能唯一确认时保留待处理状态，不必重复调用模型。

引文有局部字符错误时，需明确提供 `--quote-corrections '校正文件.json'`。
格式为 `[{"id":"R1-M1","old":"原始模型引文","new":"原稿中实际引文"}]`。
旧值必须匹配模型输出，新值必须逐字存在于原稿；每次校正会显示在审阅报告末尾。
不能用此功能悄悄改写模型意见。原始响应始终保留。

验证命令

```powershell
.\.venv-review\Scripts\python.exe -m unittest discover -s tests -p test_obsidian_review.py -v
```
