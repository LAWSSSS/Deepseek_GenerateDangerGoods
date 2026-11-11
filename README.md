# 废钢监控异物合成

当前点评试验是 [5 件 × 2 背景的 6×6 池化＋35% 薄雾组合](DISPLAY_COMBINED_VALIDATION.md)，沿用全图均光，与仅加 35% 雾和零档比较。报告全部使用完整监控画面，带漆金属积灰继续减半。十组仍为待审，批量默认继续使用回退版；上一轮[细梯度对照](DISPLAY_FINE_GRADIENT_VALIDATION.md)保留。

新增[多背景批量展示流程](DISPLAY_BATCH.md)：使用当前 44 个模型条目，按每张背景独立选位、横躺落稳、PBR 渲染和 AI 有限迭代，直接输出 PNG；命令行与网页 `/batch` 共用执行器。默认采用废钢灰强融合、表面/组合承托/局部嵌入，以及证据不足输出待审图。`accepted_display` 与旧流程正式验收分开。

最新一轮交付了 12 组固定摆位摄影对照，并验证按需渲染及局部裁剪。相同几何输入的渲染计时降低 78.4%，摄影逐图复核 1/12，扩批停在小样门槛。 [本轮效果与成本](DISPLAY_COST_VALIDATION.md)

上一轮 138 项跨背景回归输出 **138 张 PNG，2 展示合格、136 待审、0 执行失败**。另交付 41 张固定摆位摄影对照与 48 项旧拒绝任务复验（45 张有图）。 [跨背景验证记录](DISPLAY_FUSION_VALIDATION.md)

## DeepSeek Harness 与当前进度

本仓库已登记为 DeepSeek Harness 工作区 `scrap-composite`，Harness 直接打开本仓库，不存在需要同步的第二份拷贝。进入该工作区创建会话即可继续。接手顺序：先读 [项目约定](AGENTS.md)、[Harness 交接说明](DEEPSEEK_HANDOFF.md)、[进度记录](WORK_LOG.md)。每次本地提交都要在提交正文和进度记录中说明改动原因及下一步要求。

最近的小范围试跑：`outputs/固定场景-处理后模型小范围测试-20261109/`。四件单件预览的网页/Blender 投影一致性较高，但场景支撑、真实尺度、遮挡和光向仍未验证；详情见[预览总览](outputs/固定场景-处理后模型小范围测试-20261109/index.html)。这些预览没有通过正式合成验收。

四件修订样片的承托候选筛查：`outputs/四件修订-承托候选-20261109/`。程序使用固定场景缓存，为 10、13、19-1、31 自动筛选承托站点并渲染三维视角；10、13、19-1 各保留 3 个待审候选，31 在 8 个站点、4 个水平视角下为 `no_candidate`。所有候选的源可见面、遮挡、现场尺度和 Photoshop 均未验证；落稳使用整件凸包单刚体近似，部件级支撑未验证。查看[筛查总览](outputs/四件修订-承托候选-20261109/index.html)和 `summary.json`。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_support_scan.py `
  --manifest 'outputs/人工反馈修订-20261109-千斤顶去杆-v1/manifest.json' `
  --ids 10,13,19-1,31 `
  --background 'external/异物生成/input/backgrounds/2026_08_06_53_4_zhongfei1_70_zhongfei2_20_medium_10_9.jpg' `
  --out 'outputs/四件修订-承托候选-20261109' `
  --target 10:296 --target 13:371 --target 19-1:243 --target 31:475 `
  --max-sites 8 --yaws 0,90,180,270 --budget 0
```

输出目录须为新目录；`--budget` 只允许 0，场景分析缓存缺失时停止并写明原因。`--sites-only` 可先只检查承托站点和同层参照。

从完整迁移包恢复的 Windows 项目。代码在 `outputs/ps-model-controller`；原图、模型、历史结果和虚拟环境保存在本机工作区，Git 只跟踪代码、配置示例和运行说明。查看 [控制器说明](outputs/ps-model-controller/README.md)。

39 件素材可建立可续跑的三维模型库，并按场景选择单件视角。原有正式合成试产仍使用 **03 轴承座、10 千斤顶、27 减震器、30 液压缸**；模型库中的待审模型不代表已通过正式合成验收。三维结构、承托、净空、源图可见面和视觉验收的门槛独立生效。

## 本机人工放置网页（待审版本）

从项目根目录安装本地网页依赖并启动：

```powershell
Push-Location outputs/ps-model-controller/placement-web
npm ci
Pop-Location
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/placement-web/server.py
```

在本机浏览器打开 `http://127.0.0.1:8765`。选择监控底图或上传图片，再从与三维模型查看器相同的目录选择一件模型（19 号拆为 19-1…19-6）。用“拖动移动/拖动旋转”切换左键行为；切到“拖动旋转”后左键拖动旋转物体 X/Y；右键或按住 Shift 左键拖动旋转 Z；Alt+拖动调整观察视角；右上圆点同样可旋转；水平角连续 360°，俯仰角范围 -89°–89°（含仰视）；右下角方块缩放，工具栏可重置视角。物体 X/Y/Z 输入（-180°–180°）也可直接设置，独立于观察视角。选择“识别可能的遮挡”后，悬停查看分割区域，勾选位于前景的区域，并可用圆点选择局部调色的承托面。生成预览后可以调用 Photoshop 导出 PNG 和分层 PSD。页面上的“自动放置”入口启动原有位置优先流程，默认模型预算为 0；当前查看器候选多为 `manual_only`，自动放置仍只接受原模型库中具备源视图证据的条目。

网页只监听本机。对话需在启动网页的进程中配置 `PS_DEEPSEEK_KEY`，并会将缩小的背景及区域标识发送给 DeepSeek；不要把密钥粘贴到聊天或提交到 Git。单次运行采用 30000 token 预算及调用前预留；网络异常且用量未知时保留预留额度。没有密钥时仍可手动选择遮挡区域。

实时占位使用中性光，Blender 待审渲染使用同一透视相机。程序从局部场景样本估算半球环境光的亮度和色偏，仅在局部钢件线索一致时估计弱方向光。在最终像素尺寸上分别匹配明暗、色偏、锐度和颗粒，保持物体透明度与孔洞。

默认启用“固定接触参考点”：先拖动物品定位，绿色十字随之移动；之后调整方位、俯角、物体 X/Y/Z 或大小时，程序维持这个原图坐标。取消勾选可恢复围绕图片中心调整。该点是近地接触簇范围的定位参考，可能位于两只脚之间，不代表该点或装配体已落稳。网页使用真实顶点范围归一化，并与 Blender 采用相同的相机距离；生成预览时再次用 Blender 核对参考点，偏差超过 1.5 px 停止。

渲染前先提取网格接触投影并复核候选承托面，打光和调色共用该面上的同一批邻近像素。承托或取样不足时明确回退为未验证的周边采样。几何预检按网格、姿态和算法签名缓存，移动、改尺寸或调色可以复用；物体已经比背景平滑时不再额外模糊，只对已有 RGB 细节做受限调整，透明度和孔洞保持不变。[本轮千斤顶同摆位对照](outputs/固定场景-接触锁定与同面光色-20261110/同摆位新旧对照.jpg)及[检查记录](outputs/固定场景-接触锁定与同面光色-20261110/检查记录.json)仍为人工待审，不代表场景验收。

相关回归：`placement-web/checks_web.py`、`node placement-web/checks_anchor.js`。在 Playwright CLI 打开的放置页生成一张启用固定参考点的预览后，可运行 `checks_anchor_browser.js`，以 `--json` 保存结果，再用 `checks_anchor_blender.py --browser-run <该 JSON> --out <报告 JSON>` 对照实际 Blender 投影。脚本使用当前姿态并在结束后恢复，覆盖输入、拖动、转角和缩放；这会使旧预览过期，需要重新生成。以上代码路径均相对 `outputs/ps-model-controller/`。

接触斑由旋转后的网格投影得到。候选承托面须覆盖每个接触斑的至少 90%，预览才会生成限于该区域的接触阴影。与物体相交的细钢件优先显示，用户确认前景后按原图像素恢复遮挡。Photoshop 导出保留匹配前的物体、调色后物体和独立阴影图层。光向、实际落稳、现场尺度、保护区和前后顺序仍需核验，成图状态保持待审。查看[同摆位新旧对照](outputs/固定场景-人工放置融合复核-20261110/新旧融合对照.jpg)。

03 轴承座另有“六视图背面修复候选”供人工对照。该候选保留原图约束，依据未标定的参考图推测背面对称；源视图轮廓和孔洞通过检查，顶部过渡接缝、背面细节与物理姿态仍待复核。候选不替换原模型，不能用于自动放置。

03 ASAHI P210 轴承座另有依据用户六视图建立的分部件重建 v1（`bearing03-parametric-v1`，目录 `outputs/03轴承座-六视图重建-v3`）。启动网页后访问 `http://127.0.0.1:8765/review/03`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。候选保留轴承内圈/外圈、两道密封、两个紧定螺钉、顶部油嘴、ASAHI/P210 铸字、两个安装长槽、抬高的槽口凸台、底部加工垫和底部拱。参考图未标定且拼图存在横向压缩/纵向拉伸；UCP210/P210/UC210 公开尺寸比例只作合理性参照，不是对实物的测量。原模型源图检查仍有填孔，新候选的单张做锈图诊断（IoU 0.6673，1 个孔被填）未通过，均只作诊断；背面、内圈轴向偏置、紧定螺钉角度和隐藏面仍待人工复核。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、切换、灰模/材质与拖动旋转。

01 长阀杆阀门另有依据用户提供六视图建立的分部件重建候选，以及在此基础上改善金属度、粗糙度的 v7 材质版。启动网页后访问 `http://127.0.0.1:8765/review/01`，可旋转切换原模型、上一版材质、新版及灰模，并查看同一中性光下的材质对照。新模型不会替换原模型或进入自动放置。参考图没有标定，侧背面材质和不可见结构为推断；原图轮廓诊断未通过，模型真实性、现场尺度、落稳及合成均待人工检查。

02 DN200 球阀另有依据同一六视图建立的分部件重建 v1。启动网页后访问 `http://127.0.0.1:8765/review/02`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、局部检查、隔离开口检查及材质对照。候选保留三辐手轮的三个开口、12 颗剖分法兰螺栓、两端各 8 个法兰螺栓孔、球芯、底部六角盲孔和双立板支脚；隔离渲染检测到 3/9/2 个开口。原模型源图诊断仍有填孔，新候选的单张做锈图诊断也未通过，均只作诊断；参考图未标定，剖分法兰左右方向、支脚形式、球芯深孔光照、OPEN/SHUT 凸字和背面细节仍待人工复核。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成。

06 卧式不锈钢压力罐另有依据用户六视图建立的分部件重建 v2（`tank06-parametric-v1`，目录 `outputs/06压力罐-六视图重建-v2`）。启动网页后访问 `http://127.0.0.1:8765/review/06`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。候选保留三顶部卷边管口、六摆动卡箍人孔、右端上下法兰口、底部排污、双腿鞍座／加强筋／支脚长槽和两条环焊缝带；下部右端法兰口、支撑加强筋和隐藏面为参考图推断。原模型源图诊断未通过，新候选的单张做锈图诊断 IoU 0.8657、细结构召回 0.1831，也未通过，均只作诊断；左侧管口、底视排污位置和顶视支撑带在参考面板间互相矛盾，背面、孔深和装配关系仍待人工复核。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、新旧切换、灰模／材质与拖动旋转。

07 立式不锈钢搅拌罐另有按照片比例和可见结构建立的分部件重建 v2（`tank07-parametric-v1`，目录 `outputs/07储罐-六视图重建-v2`）。启动网页后访问 `http://127.0.0.1:8765/review/07`，可旋转切换原模型、新版和灰模，并查看类别参考图新旧对照、关键局部、隔离通孔、同光照材质、变光和金属反射对照。本件六视图映射为 `category_reference`：参考罐有中央人孔和顶部管口环，照片则是偏置盲盖大端口、蓝色电机支架和前左侧循环管，候选以照片为准，不套用参考顶部。候选保留三条折板支腿与调平脚、螺栓锥底法兰、下锥排污、盲盖大端口、提环通孔、视镜、循环长管、低侧阀门、电机／齿轮箱／散热筋和不锈钢支架；三腿截面、支架背面、视镜、阀门内部和隐藏面为推断。原模型源图诊断未通过（填 5 个孔、细结构召回 0.2716），新候选的轮廓诊断 IoU 约 0.820、细结构召回约 0.617，也未通过，只作诊断；顶部端口、提环和排污口的真实开口另由隔离渲染与射线检查确认。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、新旧切换、灰模／材质与拖动旋转。

08 分离过滤器另有按照片比例和可见结构建立的分部件重建 v3（`separator08-parametric-v1`，目录 `outputs/08分离过滤器-六视图重建-v3`）。启动网页后访问 `http://127.0.0.1:8765/review/08`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。六视图映射为 `visual_match_unverified`：参考图未标定，用于外观与背面拓扑，不作为测量。候选保留两个对置四螺栓法兰（每侧 4 个真实螺栓孔和中心通孔）、顶部锻造提环通孔与加强带、上／下绳状环焊缝、喷嘴角焊缝、蓝色金属漆与局部锈斑，以及黄铜自动排水器的滚花压紧环、椭圆绿色视窗和底部排污口。背面法兰／视窗、法兰厚度与孔深、排水器内部、提环加强带和隐藏面为推断；照片中未见支撑，落稳与现场尺度未验证。原模型六视角审计记录到漂浮碎片且融合了细结构，故未复用旧网格。隔离渲染检测到法兰 5 个开口、提环 1 个、排污喷管／螺母 1 个，均通过；网格封闭性、材质和金属反射检查通过。原图轮廓诊断 IoU 约 0.879（低于旧门槛 0.90，未填提环孔，细结构召回 1.0），只作诊断；候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、新旧切换、灰模／材质与拖动旋转。

09 分离过滤器 1 另有按照片比例和可见结构建立的分部件重建 v1（`separator09-parametric-v1`，目录 `outputs/09分离过滤器1-六视图重建-v4`）。启动网页后访问 `http://127.0.0.1:8765/review/09`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。六视图映射为 `visual_match_unverified`，未标定，只用于外观与背侧拓扑。候选保留顶法兰中心通孔与 8 个螺栓孔、侧法兰中心通孔与 8 个螺栓孔及侧管口小圆螺塞、顶部排气／丝堵、上部两道绳状环焊缝、筒体、铭牌（文字不可辨，未复制）、下部鼓形封头与裙座、四腿及三角加强筋、支脚长圆孔、底部自动排水器的黑色视窗／试验件、黄铜芯、蓝管、黑色弯头和方管吊架，以及红色流向标记；顶／侧法兰内孔加了深色衬套。原照是 3/4 产品图，侧法兰呈椭圆面且与对称四腿的正视投影不能同时严格成立，候选按侧管口沿 +Y、四腿对称解释。背面、四腿真实方位、支脚尺寸与长圆孔、法兰厚度／孔径／孔深／凸面、排水器型号与内部、吊架、铭牌文字、排气塞尺寸／方位和隐藏面为推断。原模型六视角审计记录源视图 IoU 0.9177、1 个填孔（未通过）；新候选原图轮廓诊断 IoU 0.8377（低于旧门槛 0.90）、1 个孔被填、细结构召回 1.0，只作诊断，不作为验收通过。隔离渲染确认顶法兰 9 个开口、侧法兰 9 个、支脚长圆孔 1 个、排水器黄铜／蓝管通孔 1 个；81 个部件网格闭合、同部件无自交，材质瓦片和金属反射检查通过。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；落稳、现场尺度和网页场景放置未验证；网页核验只覆盖加载、新旧切换、灰模／材质与拖动旋转。

10 千斤顶（10 t 立式油压千斤顶）另有按照片比例和可见结构建立的分部件重建 v2（`jack10-parametric-v1`，目录 `outputs/10千斤顶-六视图重建-v2`）。启动网页后访问 `http://127.0.0.1:8765/review/10`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。六视图映射为 `visual_match_unverified`，未标定，只用于背侧和底面拓扑。候选保留红色铸造底座（四角通孔、底面十字筋、三组前侧拱形凸台与螺栓）、筒体、镀铬柱塞、六角锁母、菱形滚花托盘、左侧提环、右侧泵体/叉头/倾斜握把插座/连杆双销、释放阀、重绘标签和独立手柄筒。原模型源图轮廓 IoU 0.945 虽通过旧门槛，但六方向审计显示左/右视压扁、手柄与主体粘连，故未复用旧网格；原照为 3/4 产品图且手柄筒悬空，候选按推断的倾斜站立姿态表示。原图轮廓诊断 IoU 0.6052（低于旧门槛 0.90）、2 个孔被填、细结构召回 1.0，只作诊断；隔离渲染确认底座四角孔 4/4、握把插座通孔 1/1、提环内开口 1/1、手柄筒通孔 1/1，网格封闭性、同部件无自交、PBR 材质和金属反射变光检查通过。四角孔的真实用途/孔深、底座背面凸台、泵内结构、插座深处、标签原始印刷和隐藏面为推断；落稳、现场尺度和网页场景放置未验证。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、新旧切换、灰模/材质与拖动旋转。

11 千斤顶1（QL5 绿色螺旋千斤顶）另有按照片比例和可见结构建立的分部件重建 v3（`jack11-parametric-v1`，目录 `outputs/11千斤顶1-六视图重建-v3`）。启动网页后访问 `http://127.0.0.1:8765/review/11`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。六视图映射为 `visual_match_unverified`，未标定且各面板不是一致转台：参考把杠杆画成圆管，照片显示为空心矩形杆；参考正视把提环画在筒体右侧，照片显示提环与杠杆同在可见侧，候选以照片为准并把参考该处视为镜像。候选保留绿色铸造底座（两级裙边、五条底面放射筋、中心凸台与盲凹）、锥形筒体、12 面绿色上环、镀铬螺杆与深色顶帽、5T 铸字、重绘标签、空心矩形操作杆及开口端、圆形插座、齿轮座圆端面与六角螺栓、下部支架/连杆、亮绿拉钮、提环与两处连接座/销。原模型源图轮廓 IoU 0.955 虽通过旧门槛，但六方向审计显示表面糊化、标签不可辨、提环/齿轮座与筒体粘连，故未复用旧网格。新候选严格原视角轮廓诊断 IoU 0.2224（照片为 3/4 而严格相机为正视，只作诊断）；315° 转台按包围盒对齐后的诊断 IoU 0.6858，也不是验收通过。隔离渲染确认杠杆矩形通孔 1/1、提环内开口 1/1；49 个部件网格闭合、同部件无自交、PBR 材质瓦片和金属反射变光检查通过；无头浏览器核验页面加载、新旧切换、灰模/材质和拖动旋转通过。齿轮座内部、拉钮功能/轴、提环背面连接、5T 浮雕、标签原始印刷、底面细节和隐藏面为推断；落稳、现场尺度和网页场景放置未验证。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成。


12 千斤顶2（2 吨卧式液压千斤顶）另有按照片比例和可见结构建立的分部件重建 v1（`jack12-parametric-v1`，目录 `outputs/12千斤顶2-六视图重建-v3`）。启动网页后访问 `http://127.0.0.1:8765/review/12`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。六视图映射为 `visual_match_unverified`，未标定且各面板不是一致转台：正视/背视把手柄画成竖立，左右视面板和原照显示水平操作位置，候选以原照为准，并把手柄水平姿态标为推断。候选保留红色焊接车架与侧板压筋、两个前大轮与贯穿轴、两个后万向脚轮和红色安装耳、锥形顶升臂与黑色顶升头、滚花鞍座、黑色泵盖与重绘 2 TON 标签、黄色 U 形提环、手柄插座和红色手柄/黑色握把、顶升臂侧面的长条警示标签。泵内结构、插座锁定机构、焊缝、轮胎花纹、手柄长度、标签原始印刷与隐藏面为推断。原模型源图轮廓 IoU 0.949 虽通过旧门槛，但六方向审计显示表面糊化、车轮/手柄/泵体粘连、标签不可辨，故未复用旧网格；其对照渲染为比较目的旋转了 -90°，库文件未改动。原图轮廓诊断 IoU 0.3569（严格相机沿手柄轴，照片为 3/4 视角，只作诊断）；24 个转台方向中按包围盒对齐的最优诊断 IoU 0.5282，也不是验收通过。隔离渲染确认黄色 U 形提环拱下通透、前轮毂通孔 1/1、手柄插座内孔 1/1；79 个部件网格闭合、同部件无自交，PBR 材质瓦片和金属反射变光检查通过；无头浏览器核验页面加载、新旧切换、灰模/材质和拖动旋转通过。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；落稳、现场尺度和网页场景放置未验证。
13 密封胶罐（两只白色塑料罐）另有按照片比例和可见结构建立的分部件重建 v1（`sealant13-parametric-v1`，目录 `outputs/13密封胶罐-六视图重建-v2`）。启动网页后访问 `http://127.0.0.1:8765/review/13`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离结构/开口、同光照材质、变光和表面反射对照。六视图映射为 `category_reference`：参考图是单只干净罐，原照是两只使用过的罐，数量不同且未标定，参考只用于罐盖/罐身/底面拓扑。候选保留两只独立罐身、轻微锥度与底部圆角、外伸罐盖与双道盖缝、盖顶外缘凸边与浅凹面、每罐双段式卡扣（前罐卡扣以侧面轮廓出现，后罐卡扣朝近 +X）、每罐两只底部小凸脚、中央浇口和简化回收式标志，以及白色塑料污渍/划痕 PBR。原模型源图轮廓 IoU 0.9775 虽通过旧门槛，但六方向审计显示表面纹理粗糙、两罐粘连、盖缝与卡扣不清，故未复用旧网格。新候选严格原视角轮廓诊断 IoU 0.8547（未通过旧门槛，只作诊断）；单罐类别参考诊断 IoU 约 0.80–0.86；按包围盒对齐的 24 方向转台最优诊断 IoU 0.9145，虽达到旧数值门槛但未标定，不作为验收通过。隔离渲染确认两只罐顶视分离 2/2、单罐六向轮廓封闭孔 0/0，两罐罐身最近顶点间距约 0.0237 m；30 个部件网格闭合、同部件无自交；白色塑料与深色标志贴图的金属度均为 0，两组主光下介质塑料高光发生移动。本件没有可确认的金属零件，金属反射不适用，不能据表面反射对照声称金属检查通过。罐间距/朝向、盖内配合、壁厚、卡扣机构、底脚功能、回收标志的准确字形/深度、污渍分布和隐藏面为推断；落稳、现场尺度和网页场景放置未验证。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、新旧切换、灰模/材质与拖动旋转。

从本地六视图重建该候选（输出目录须为新目录；相同签名的完整结果可复用）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item.py --number 1 --reference "$env:USERPROFILE/Desktop/六视图/工业阀门六视图目录图.png" --out 'outputs/01阀门-六视图重建-v7'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_01.py
```

09 复现命令（输出目录须为新目录；入口按已登记哈希查找六视图；`checks_rebuild_09.py` 当前校验已生成的 v4 记录）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_09.py --number 9 --out 'outputs/09分离过滤器1-六视图重建-v5'
```

10 复现命令（输出目录须为新目录；入口按已登记哈希查找六视图；`checks_rebuild_10.py` 当前校验已生成的 v2 记录）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_10.py --number 10 --out 'outputs/10千斤顶-六视图重建-v3'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_10.py
```

11 复现命令（输出目录须为新目录；入口按已登记哈希查找六视图；`checks_rebuild_11.py` 当前校验已生成的 v3 记录）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_11.py --number 11 --out 'outputs/11千斤顶1-六视图重建-v4'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_11.py
```

12 复现命令（输出目录须为新目录；入口按已登记哈希查找六视图；`checks_rebuild_12.py` 当前校验已生成的 v3 记录）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_12.py --number 12 --out 'outputs/12千斤顶2-六视图重建-v4'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_12.py
```

13 复现命令（输出目录须为新目录；入口按已登记哈希查找六视图；`checks_rebuild_13.py` 当前校验已生成的 v2 记录）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_13.py --number 13 --out 'outputs/13密封胶罐-六视图重建-v3'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_13.py
```

输出含可编辑 Blender 文件、GLB、2048 px 材质图集、六方向与同光照材质对照、24 张转台视图及 JSON 检查记录。v7 默认读取保留的 v6 模型制作同光照对照，需保留其输出目录。四个手轮开口和八个底部安装孔通过独立部件透明度检查；同部件使用 BVH 检查非相邻面交叠。检查不能证明未观察部分或部件之间的装配关系。01、02 各有独立入口，其他物品须逐件建立对应结构与参考。

39 件的中性光六方向检查图与逐件记录位于 `outputs/三维模型-39件六视角审计`。网页提供 08 分离过滤器与 19 气瓶组的漂浮碎片修复候选、29 减震器的源图孔洞修复候选，以及 15、16、36 号喷罐的圆柱截面修复候选。源图检查通过后才显示为可选，但隐藏结构、部件关系和落稳仍待审，原模型保留。其他模型的缺陷不因六方向渲染而自动视为修复。

实时占位为金属材质提供中性反射环境。插入时从多个方向的细钢件明暗尝试估计低置信度光向，线索不足时使用渐变半球环境光和弱顶光。细钢件使用原图尺度的线段候选与受限分割，前后关系仍需确认。

离线回归（不调用模型）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/placement-web/checks_web.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_mesh.py
```

## 三维模型库与场景选角

在项目根目录运行（第一次重建会耗时，重复命令从逐件记录续跑）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_library.py --retry
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_select_view.py --view-only --asset 3 --background '现场底图.jpg' --out 'outputs/选角结果' --target-px 224.6
```

模型库按来源哈希复用已有模型，逐件记录原视角轮廓、孔洞和细结构；做锈素材中的孔洞仅在原貌素材的透明区域提供对应依据时尝试一次受限修复，失败时保留原始网格和拒绝原因。单张图片中不可见的表面及孔深属于推断，关键结构仍需复核。

`--view-only` 保留原先只输出视角的诊断流程；它读取场景的深度、承托区、保护区和俯角，比较落稳且源图可见面不少于 80% 的姿态，并复核俯角区间端点。`selected-view.png` 是独立视角而非已验收合成；`result.json` 记录依据及待审项。未提供 `--target-px` 时使用画面比例产生候选尺寸，该值不代表现场实测。缺少所需证据或没有合格候选时停止并记录原因。

场景分析复用按背景和配置签名的缓存；新场景没有缓存时需要配置视觉模型和本机密钥，并传入 `--budget`（最高 30000）。默认预算为 0，不会调用外部模型。

## 先选位置、再选视角和调色

`mesh_select_view.py` 默认使用位置优先流程，也可直接运行 `mesh_location_first.py`。程序先根据场景承托区、净空、保护区及局部参照排序接触位置，再在每个固定位置检查真实三维视角。候选还需通过落稳、源图可见面、各接触斑、实际轮廓和俯角端点检查；不合格时自动尝试下一个位置。确定位置和视角后才从同一承托面取样调色，排除前景、阴影和保护区，并验证 Photoshop 实际导出。

```powershell
$scene = Get-Content 'outputs/固定场景-39件尺寸标注/场景配置.json' -Raw | ConvertFrom-Json
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_location_first.py `
  --asset 03 --background $scene.background --target-px 224.6 `
  --settled-view 'work/03-smooth-poses/pose-45' `
  --previous 'outputs/轴承座-圆润后场景检查-v4' --out 'outputs/位置优先-新运行' --budget 0
```

`--settled-view` 可选；仅在网格完全一致且单部件落稳通过时复用物理结果，程序会重新投影接触点。省略时会从当前模型重新模拟。`--previous` 可选，只用于输出对照图，并检查其物品与背景。每次使用新输出目录。结果包括分层 PSD、全图 PNG、对照图、位置与视角诊断；估算尺度和缺少的结构、视觉复核仍会标为待审。

## 六视图参考与圆润处理

六视图拼图按物品外观关联，关联清单位于 `outputs/ps-model-controller/six-view-map.json`。参考图没有相机标定，部分是相似类别或包含多个物品，不能直接作为精确几何。程序记录这些不确定性；目前使用它们进行视觉对照，并未进行多视图重建或补齐不可见结构。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_smooth_library.py `
  --six-views '<六视图文件夹>' --out 'outputs/三维模型库-圆润复验'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_smooth_report.py
```

平滑限制顶点位移，并保护源视图边缘；输出前检查面翻转、连通性、纹理、体积及源视图中的轮廓和孔洞。未通过保护门槛时保留原模型。局部起伏指标仅用于量化变化，不表示外观或隐藏结构已经验收。使用圆润模型进行场景选角时，给 `mesh_select_view.py` 增加 `--library 'outputs/三维模型库-圆润复验/results.jsonl'`；几何发生变化后应重新验证落稳，不复用旧的刚体姿态。

轴承座示例可通过 `mesh_sixview_review.py` 将同一组近似方向上的平滑前后渲染与参考拼图并列。图中 IoU 是新旧模型之间的轮廓一致性，不是对参考图的精确匹配。使用 `mesh_pipeline.py --inspection-reference <上一版/result.json>` 可以保留旧图作为对照，同时允许重新搜索实际接触点；`--inspection-baseline` 则继续锁定原接触点。

在工作区根目录打开 PowerShell，使用已经安装的环境和减面结果运行：

```powershell
. .\enter_handoff.ps1
$py = 'work/triposr-env/Scripts/python.exe'
$code = 'outputs/ps-model-controller'
$mcp = 'work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe'
$blender = 'work/blender45/blender-4.5.3-windows-x64/blender.exe'

$common = @(
  '--build', 'outputs/固定场景-投影孔洞减面复验',
  '--scene', 'outputs/固定场景-39件尺寸标注/场景配置.json',
  '--approved-from', 'outputs/固定场景-空间姿态试产',
  '--weights', 'work/v2-models',
  '--cache', 'work/v2-cache',
  '--blender', $blender,
  '--mcp', $mcp,
  '--render-device', 'OPTIX'
)
& $py "$code/mesh_pipeline.py" @common --out 'outputs/固定场景-朝向与承托复验' `
  --geometry-only-from 'outputs/固定场景-三维支撑关系终测' --budget 0
```

正式复测需要在运行它的本机进程中提供 `PS_DEEPSEEK_KEY`，启动 Photoshop，然后执行：

```powershell
& $py "$code/mesh_pipeline.py" @common --out 'outputs/固定场景-三维续测' --retry
```

复测报告在各输出目录的 `index.html` 和 `token-usage.csv`。本地诊断不会调用 DeepSeek 或交付合成图片。正式流程只对通过必要检查的候选使用 Photoshop，并将真实合成预览交给 DeepSeek 复核。

## 三维 v2 验证

可见面按逐像素深度和源视图表面回投计算；256、512 两种分辨率相差超过 2 个百分点时结果为未知。网格部件使用均匀密度体积质心与相对质量；非封闭网格、装配关系变化和未通过支撑的姿态保留待审。场景候选先筛选可用尺度参照，实际移动后再次核验尺度。正式流程要求分阶段结构复核，历史 v1 模型结论不会自动视作 v2 通过。

`--geometry-only-from` 仍是离线姿态诊断入口：需要同一背景的历史场景俯角，设 `--budget 0`，不会运行结构模型、场景选位或 Photoshop。输出中的 `checks` 记录已完成检查与未知项。正式复测仍需要本机进程环境变量 `PS_DEEPSEEK_KEY`，并以 `--budget 30000` 限制本轮模型用量。

正式批处理中的自动结构缺陷修复、独立阶段诊断开关和物理与渲染分离缓存仍待接入。

## 03 轴承座单件检查图

轴承座检查流程会比较 48k、96k 面网格，对源图孔沿和外轮廓进行保护，生成 2048 像素纹理，并从素材提取受限的表面细节和微小起伏。不可见表面的纹理仍是投射推断。正式尺寸参考仍按来源标记为估算或测量。人工检查图可在参照不足时标记待验，正式流程仍要求完整尺度与模型复核。

在前面的 PowerShell 设置之后，可从同一批次重新生成网格和姿态，再输出到一个新目录：

```powershell
$prior = Get-Content 'outputs/固定场景-v2最终几何复测/results.jsonl' -First 1 | ConvertFrom-Json
$camera = 'work/03-source-camera.json'
$prior.reconstruction.source_camera | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $camera
$alpha = Join-Path (Split-Path $prior.source_view) 'source-alpha.png'
$detail = 'work/03-detail'
$poses = 'work/03-poses'
& $py "$code/mesh_detail.py" --mesh $prior.pre_optimization_mesh --alpha $alpha --camera $camera --asset $prior.rust --out $detail --blender $blender --render-device OPTIX
if ($LASTEXITCODE -ne 0) { throw '网格细节处理失败' }
& $py "$code/mesh_prepare_pose.py" --mesh "$detail/mesh-texture/mesh.glb" --source-view "$detail/mesh-texture/source-view.png" --alpha $alpha --camera $camera --out $poses --blender $blender --render-device OPTIX
if ($LASTEXITCODE -ne 0) { throw '姿态检查失败' }
& $py "$code/mesh_pipeline.py" @common --out 'outputs/轴承座-新检查' --budget 0 --inspection-item 03 --inspection-pose "$poses/pose-45" --inspection-source-view "$detail/mesh-texture/source-view.png"
```

输出包含全图、分层 PSD、局部与结构对照，以及记录接触位置、视角、尺度依据和光色指标的 `result.json`。

仅更新材质时，可用 `mesh_retexture.py` 重新投射素材；它会核对导出前后的顶点和面以及源视图轮廓。`mesh_render_pose.py` 只允许几何完全相同的网格复用已验证落稳姿态。检查入口的 `--inspection-baseline <上一版/result.json>` 固定已验证的接触锚点和尺寸，生成可直接比较的检查图。
