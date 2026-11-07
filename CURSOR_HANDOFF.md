# Cursor 交接：逐件完成 02–39 号三维模型

## 用户目标与当前授权

- 用户已确认 **01 号阀门的新版金属外观通过**。以 `outputs/01阀门-六视图重建-v7` 为材质与细腻度的参考，保留 01 的原版和 v6。
- 接下来从 **02 到 39 按编号逐件检查并修复或重建**，继续运行，不必每件都停下等待用户。每件保存可检查的结果；遇到证据不足时标注待审并继续其他物品，不得声称已验收。
- 当前工作重点是各物品本身的形体、细节和金属质感。模型应能进入现有人工放置页面供检查；正式自动放置、落稳、现场尺度、细钢筋遮挡与环境光融合仍需各自验证，不能因为 01 的外观获准而标为通过。
- 用户要求尽量使用他提供的六视图；对未标定或与原物品不完全一致的参考，标出推断并优先保留可观察的关键结构。不足的信息可查找厂家等可靠资料，记录来源，不能用相似型号冒充精确实物。

## 项目和可用资料

- **仓库与工作目录：** `D:\CodexProjects\experiments\scrap-composite`，当前基线提交 `02340b3`。从这里打开 Cursor；桌面“废钢合成-完整迁移包”不是当前 Git 仓库。
- 用户六视图：`C:\Users\zhaoc\Desktop\六视图`；逐件参考映射：`outputs/ps-model-controller/six-view-map.json`。这些图是外观参考，并非已标定的摄影测量数据。
- 39 件原图：`external/异物生成/input/objects`；对应高清原貌和做锈素材：`outputs/物品高清输入-39件/原貌HD`、`生锈HD`。原模型与相机记录：`outputs/三维模型库-圆润复验/results.jsonl`。
- 六方向原模型审计及逐件记录：`outputs/三维模型-39件六视角审计/index.html`、`audit.jsonl`。已有 08、15、16、19、29、36 号等局部修复候选：参见根目录 `README.md`。不要把候选存在视为该物品整体通过。
- 01 示例入口与工作方式：`outputs/ps-model-controller/mesh_rebuild_item.py`、`mesh_rebuild_item_worker.py`、`valve01_config.json`。**当前入口只支持 01，不能把它当作支持其他物品的通用建模器。** 本机 Blender：`work/blender45/blender-4.5.3-windows-x64/blender.exe`；Python：`work/triposr-env/Scripts/python.exe`。
- 本机网页代码：`outputs/ps-model-controller/placement-web`，当前服务 `http://127.0.0.1:8765/`；01 检查页为 `/review/01`。页面候选与原模型分开，候选保留 `manual_only`，避免进入未经验证的自动流程。

## 对每件的工作要求

1. 读取对应原图、做锈图、参考映射、旧网格及六方向审计；明确关键孔、细杆、附件、材质和已知变形。先判断保守修复或分部件重建更合适，不统一套用阀门形状或用平滑掩盖缺失结构。
2. 使用与实物相符的金属、锈蚀、蓝漆／其他漆面、塑料或橡胶材质；参照 01 的 PBR 金属度、粗糙度和中性反射环境，保留细腻纹理。高光必须随灯光／视角变化，不把固定灯光和阴影烘进底色。用户已认可 01 外观，不代表 39 件都应使用同一金属参数。
3. 保持原模型及素材；每件使用独立版本目录和可复现配置，输出 GLB、可编辑 Blender 文件、贴图、六方向／关键细节新旧对照、结果 JSON 与证据来源。引用已有候选时验证源和输出哈希。模型和贴图等大文件放在已有忽略的 `outputs`／`work`，不要提交。
4. 对真实孔隙、细结构、关键附件和已知零件关系做针对性检查；同部件网格检查及原照片轮廓、孔洞、细结构的旧门槛继续记录。六视图未标定时，不把近似轮廓分数称为实测精度。无法证实的背面、尺寸、孔深、装配关系和场景姿态一律标待审。
5. 在网页中按物品和版本展示候选，允许旋转、查看材质／灰模和对照；不能覆盖旧会话或把未经验证的候选自动送入正式合成。核对 Three.js 与 Blender 的坐标、尺寸、方向、材质和投影一致性。现有旧模型采用特定坐标约定，须兼容原路径。
6. 每件完成后写入可续跑的进度记录，至少包含编号、输入／输出哈希、方法、参考、可确认与推断内容、检查结果、待审原因及文件路径。中断后从下一个未完成或未能检查的编号继续，最终整理 02–39 的总索引。

可按类别复用基础组件和验证流程，但每件单独核对形体；不要一次批量生成 38 个外观相同、未经检查的模型。若外观信息不足导致无法达到质量目标，保留最佳候选、原因和下一步所需证据，再处理下一件。

## 验证与 Git 约定

从仓库根目录运行与改动适配的检查，至少保留并在需要时扩展：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/placement-web/checks_web.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_mesh.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_v2.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_01.py
```

实际用浏览器和 Blender 检查新版本，不只依赖单元测试；记录各次检测范围。不要声称运行了不存在的模型、视觉复核或 Photoshop 验收。

修改前检查 `git status`；每完成一个可验证的相关代码阶段，仅暂存本任务代码并创建本地提交，格式为 `feat: 简短中文说明`、`fix: 简短中文说明` 等。无须修改任何 Git 配置；若本机缺少提交身份，可在单次 `git -c user.name=Codex -c user.email=codex@localhost commit ...` 中临时指定。不得提交凭据、密钥、非示例 `.env`、无关改动或大型生成文件；不得 push、创建远程仓库／PR、重写历史、`reset --hard`、`git clean` 或批量删除。

不要在文档、日志或对话中粘贴密钥；任何外部模型只能使用本机已有的安全配置。用户指定把任务交给 Cursor 执行，请从 02 开始实际工作，保持可续跑并持续推进到 39。
