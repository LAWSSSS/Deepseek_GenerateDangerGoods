# 废钢场景合成

将工业物品的三维模型放入废钢监控背景，输出原尺寸 PNG、几何检查和视觉复核记录。批量运行、手动摆放和 Photoshop 导出可独立使用。现场尺度、承托或遮挡证据不足时，结果标为待审。

| 目录 | 内容 |
| --- | --- |
| `src/scrap_composite/` | 公共数据协议、凹处判断、摄影复核、参数库与 Photoshop 适配 |
| `tools/ps-model-controller/display_batch/` | 批次执行、场景分析、尺度与接触、材质、渲染缓存及合成 |
| `tools/ps-model-controller/placement-web/` | 本地选位、模型查看和批量页面 |
| `tools/ps-model-controller/` | 命令行入口、Blender 工作进程、模型重建、配置示例与专项回归 |
| `tests/` | 公共包的离线回归 |

## 安装与运行

建议使用 Windows 和 Python 3.12。从仓库根目录执行以下命令。Blender、模型和本地数据的目录约定见[运行说明](docs/runtime.md)。

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -e '.[photoshop,photography]'
& .venv/Scripts/python.exe -m unittest discover -s tests
```

在配置好本地素材、Blender 和模型权重后，复制批量配置来运行。配置中的输入、输出及缓存目录相对于仓库根目录。首次可用 1 个任务进行检查。

```powershell
New-Item -ItemType Directory -Force work | Out-Null
Copy-Item tools/ps-model-controller/display-batch.example.json work/batch.local.json
# 编辑 backgrounds、assets、count、render_device 和 provider_config。
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/batch_display.py run --config work/batch.local.json
```

网页需要 Node.js。安装前端依赖后，启动本地服务并打开 [批量页](http://127.0.0.1:8765/batch)。

```powershell
npm ci --prefix tools/ps-model-controller/placement-web
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/placement-web/server.py
```

视觉服务的地址与模型通过配置设置。批量使用独立的 API 凭据，标准百炼示例为 `batch-config.qwen.example.json`。Token Plan 用于[交互诊断](skills/token-plan-vision/SKILL.md)。凭据由进程环境或本机凭据存储提供。

批次会在指定输出目录存储原尺寸 PNG、`index.html`、`results.jsonl` 和实际请求用量。相同配置和输入可续跑；改变算法或输入后使用新目录。`realism_v1` 是可选的材质与成像试验预设。

逐件模型修复、结构化摄影诊断和 Photoshop 实际回读见[写实工作流](docs/model-realism-workflow.md)。`component_v1` 为独立候选试验；先检查模型，再确认是否进入场景测试。
