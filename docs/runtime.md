# 本地运行依赖

源码可安装为 Python 包。实际场景生成需要本地素材、模型权重和 Blender。以下路径均相对于仓库根目录。

| 本地路径 | 用途 |
| --- | --- |
| `external/backgrounds/` | 示例批量配置使用的背景目录，可替换为自己的目录 |
| `references/六视图/` | 手工参数化重建用的六视图 |
| `external/textures/wear-sample.jpg` | 旧 Photoshop 三维流程的做旧纹理，使用有权复用的本地 JPEG |
| `work/triposr-env/` | 本地 Python 3.12 渲染环境 |
| `work/v2-models/` | MobileSAM、深度模型和 BiRefNet 权重。三维重建还需要 TripoSR |
| `work/blender45/blender-4.5.3-windows-x64/` | 批量渲染使用的 Blender 4.5.3 和 OCIO 配置 |
| `outputs/三维模型库-圆润复验/results.jsonl` | 本地模型库索引，指向模型、物品照片及已生成的衍生资产 |
| `work/direct-deepseek/sequence.json` | 本地模型查看索引 |
| `outputs/固定场景-39件尺寸标注/场景配置.json` | 交互工作台的本地场景配置 |
| `work/render-materials/nearby-worn-steel-v15/pack.json` | `realism_v1` 默认材质包清单，可在 effects 中指定自己的清单 |

渲染环境使用控制器目录下的 `requirements-v2.txt` 和 `display-batch-requirements.txt`。依图形设备设置 `render_device` 为 `CPU`、`CUDA` 或 `OPTIX`。安装 Bootstrap 所需资源时，`setup_v2.ps1` 会校验分割和深度权重。已有 Windows 迁移包可以用 `setup_handoff.ps1` 初始化路径与共享环境，安装步骤需要该包的 `vendor/` 和 Blender 文件。

主要入口和相对依赖如下。

- `batch_display.py run` 执行批量 PNG 合成。`batch_display.py diagnose` 读取同一配置做离线诊断。
- `placement-web/server.py` 启动本地工作台。`placement-web/review_viewer.py` 单独浏览本地模型库。
- `python -m scrap_composite.photoshop_cli --help` 查询 Photoshop 导出选项。需要 Windows Photoshop 和本机 Photoshop MCP。
- `python -m scrap_composite.knowledge_cli --help` 查询本地参数资料库的录入和查询选项。
- `interactive_token_plan.py` 处理对话发起的视觉诊断，接收明确指定的 1 至 4 张图。使用本机配置的 Token Plan 凭据。

维护时在根目录运行检查。下例验证核心协议与本轮写实策略，不调用在线视觉服务。含 Blender 验证的检查需要上表中的 Blender。

```powershell
& .venv/Scripts/python.exe -m unittest discover -s tests
& work/triposr-env/Scripts/python.exe -X utf8 -B -m unittest discover -s tools/ps-model-controller -p 'checks_realism*.py'
node tools/ps-model-controller/placement-web/checks_anchor.js
```
