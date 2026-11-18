# 建模与生成工作台

主入口是 `python -m scrap_composite`。网页、命令行与旧脚本使用同一份资产目录和批次服务。新代码位于 `src/scrap_composite`。

| 模块 | 负责的工作 |
| --- | --- |
| `data` | 读取历史模型目录和复核记录，维护建模任务、发布版本、背景选择及批次快照 |
| `modeling` | 登记素材、执行重建或导入、应用已验证的部件材质、回读模型和网页代理 |
| `generation` | 编排队列和缓存，处理场景分析、物理摆放、渲染、合成与实际导出。视觉审核和反馈分别位于 `gate/review.py` 与 `gate/feedback.py`，Photoshop 分流位于 `execution.py` |
| `frontend` | 提供建模、批量、手动摆放和结果页面。网页通过服务接口读取目录、提交任务和发布候选 |

共用协议在包根目录。实际 Photoshop 操作通过固定 JSX 和本机 MCP。Blender、CPU TripoSR 和旧试验脚本沿用项目的运行环境。旧 `display_batch` 导入名转发到对应模块。

## 建模与发布

在本地工作台打开 `/modeling`。选取已有物品或填写名称，上传照片、透明图、GLB 或 Blender 文件。导入多个网格的 Blender 文件时，用 `object:名称` 或 `collection:名称` 指定主体。导入的 Blender 模型需具有可读、已打包的贴图。程序保存当前评估几何和材质，生成网页 GLB，并检查重新打开的文件和代理边界。

六视图按方位分别上传并呈现在候选对照中。重贴图可以选择与源 GLB 散列和部件匹配的现有材质库。带物理尺度的贴图需要长轴毫米数。复杂或不可见的结构由预览复核，用户提供的长度保留来源与估计状态。支持同件重新导入或重建、重贴图、重做 Blender 验证，以及从失败任务继续。

```powershell
python -m scrap_composite model build --name "新物品" --source "D:/素材/物品.png" --view "front=D:/素材/正面.png"
python -m scrap_composite model build --asset 40 --source "D:/素材/新候选.blend" --subject "collection:Assembly"
python -m scrap_composite model status <任务ID>
python -m scrap_composite model resume <任务ID>
python -m scrap_composite model publish 40 <已预览的任务ID>
```

`work/modeling` 存放素材、候选和 SQLite 目录。每个候选有单独目录及模型、预览和核验散列。发布更新当前版本指针，已提交的批次使用提交时固定的模型版本与图像散列。生产批量、手动摆放和查看器选择器包含已发布模型。历史源目录、已确认修订及可选物品规则来自兼容读取层。

## 批量合成与 Photoshop

```powershell
python -m scrap_composite generate run --config work/batch.local.json
python -m scrap_composite generate run --config work/batch.local.json --photoshop-mode on_issue
python -m scrap_composite web
```

批量页允许选背景与已发布物品。旧配置仍使用 `backgrounds` 目录，`background_files` 可以按该目录中的文件名缩小范围。固定任务保留现有 `manifest` 格式。第一次提交写出 `input-snapshot.json`，保存模型版本、背景和输入散列。图片或模型文件被修改时暂停，恢复对应内容后可续跑。需要改变任务时使用新的输出目录。

| `photoshop_mode` | 行为 |
| --- | --- |
| `off` | 默认合成 PNG 并审核 |
| `always` | 程序检查通过的草稿先导出 PNG 和 PSD，再审核实际 PNG |
| `on_issue` | 先审核普通 PNG，有明确可执行的图层调整且剩余轮数时调用 Photoshop |

两个 Photoshop 模式启用受保护图层。旧 `prepare_photoshop` 仍可单独准备独立工具使用的图层。执行模型的绝对参数绑定图片、操作单位和可用证据。一次有效 Photoshop 调整计入同一修正轮数，导出结果重新复核。几何问题回到候选校验，材质和反射交给 Blender。Photoshop 失败时保留可恢复记录，全局故障显示 `paused_photoshop`。PSD 下载链接、实际回读、渲染及导出耗时和 API 用量出现在结果记录中。

历史报告和旧配置可继续读取。迁移前的批次队列绑定其原实现指纹，在相应实现版本内续跑。新版批次以输入快照及实现指纹续跑。纯摄影调整沿用当前渲染通道，发布新模型后相关几何与渲染按新散列生成。
