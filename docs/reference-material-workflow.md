# 六视图参考材质

六视图用于学习磨损的外观和分布，不能直接作为校准后的 PBR 或六台相机的摄影测量。参考图中的白底、分格、文字、圆柱明暗和零件形状不能进入表面纹理。原模型、原贴图和生产默认保持不变，候选单独导出为打包依赖的 Blender 文件。

## 按六视图里的锈斑位置重贴

01 阀门、25 蓝色液压缸和 33 灭火器有对应的物品六视图，可使用 [部件对位配置](../tools/ps-model-controller/six-view-oxide.example.json)。清单绑定图片和模型散列。各视图的筒身区间、横截面轮廓与遮挡区域在配置中记录。圆柱图按角度合成，手轮、端面和压把等用平面图。匹配点和坐标轴使用渲染归一化后的网格，Blender 导入坐标需要经过工作进程的轴向转换。

- 输出图集的三个数据通道依次是锈层或污渍、露钢、参考支持度。颜色通道取自参考锈斑，压低大范围明暗差。已有粗糙度、微凹凸和金属材质分区继续参与受光。灭火器塑料底座的棕斑作为非金属污渍。
- 显眼印字、软管遮挡、孔洞和不对应的端部从源取样中排除。源模型印字仍由原几何或 UV 保护。斜视取样随掠射角降低，筒体投射在端盖法向上衰减。支持度不足时保留较轻的基础磨损，模型的隐藏面位置仍未知。
- 25 的 `Right` 参考面有不同的端部件，另一面真实可对应的法兰圈仍可对位。01 的 `Right` 仅采用能核对的局部区域。对照相机匹配方向和大致俯角，生成参考图之间仍存在结构和光线差异。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/prepare_six_view_oxide.py --spec tools/ps-model-controller/six-view-oxide.example.json --references <六视图目录> --baseline-library <分件PBR库.json> --output work/six-view-oxide
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/diagnose_reference_materials.py --reference-manifest work/six-view-oxide/library.json --output outputs/six-view-models --resolution 1024 --pose-mode inspection --surface-dust .24
& work/blender45/blender-4.5.3-windows-x64/blender.exe -b --factory-startup --python-exit-code 1 --python tools/ps-model-controller/render_six_view_review.py -- --registration tools/ps-model-controller/six-view-oxide.example.json --candidates outputs/six-view-models --output outputs/six-view-models/registered-views
```

降采样预检使用 `--resolution 512`。`render_six_view_review.py --baseline <上一版候选目录>` 会按同样的相机和灯光增加每件的上一版正面。候选重开校验包含所有对位图集的散列、材质连接以及源几何和 UV 的指纹。

### 局部细网格

`refine_six_view_meshes.py` 从已验证的六视图 `.blend` 生成独立几何候选。[示例配置](../tools/ps-model-controller/six-view-mesh-detail.example.json) 绑定零件、归一化坐标和原视图散列。01 增加连续螺纹，25 以实测模型孔径重做两侧钢衬套的细圆弧和唇口，33 缩短筒肩并增加从参考正面取表盘的压力表。细分与补件限定在显式部位。 [Blender BMesh 局部细分](https://docs.blender.org/api/4.5/bmesh.ops.html#bmesh.ops.subdivide_edges)

```powershell
& work/blender45/blender-4.5.3-windows-x64/blender.exe -b --factory-startup --python-exit-code 1 --python tools/ps-model-controller/refine_six_view_meshes.py -- --baseline outputs/six-view-models --recipe tools/ps-model-controller/six-view-mesh-detail.example.json --output outputs/six-view-fine-mesh
& work/blender45/blender-4.5.3-windows-x64/blender.exe -b --factory-startup --python-exit-code 1 --python tools/ps-model-controller/render_six_view_review.py -- --registration tools/ps-model-controller/six-view-oxide.example.json --candidates outputs/six-view-fine-mesh --baseline outputs/six-view-models --detail-recipe tools/ps-model-controller/six-view-mesh-detail.example.json --output outputs/six-view-fine-mesh/review-views --views Front Left Top
```

检查比对未编辑面的位置和 UV，回填其原法线，重开后核对贴图像素来源和原材质图片字节。两处衬套都用穿孔射线验证，新增表盘视线用局部射线检查原阀头遮挡。螺距、筒肩曲线和表壳尺寸为外观估计。几何候选用于外观点评，现场接触和投影需另行复验。

### 微增锈与两场景检查

微调在已经对位的铁锈通道上完成。默认将覆盖权重 $m$ 增加 $0.24m(1-m)$，单处最多增加 0.06。露钢、原有贴图与细网格仍由上一轮候选提供。新目录保存三份打包模型、通道读回、源模型及候选散列。实景投影和刚体接触使用打包文件中的同一网格。渲染只追加该物品及材质，照明和接收面根据监控照片重建。

```powershell
$blender = 'work/blender45/blender-4.5.3-windows-x64/blender.exe'
$python = 'work/triposr-env/Scripts/python.exe'
$controller = 'tools/ps-model-controller'
& $blender -b --factory-startup --python-exit-code 1 --python "$controller/boost_six_view_oxide.py" -- --baseline <细网格候选目录> --output <增锈候选目录>
& $python -X utf8 -B "$controller/validate_six_view_scenes.py" run --out <实景试验目录> --backgrounds <背景目录> --candidates <增锈候选目录>/asset-candidates.json
& $python -X utf8 -B "$controller/validate_six_view_scenes.py" retry-surface --out <实景试验目录>
& $python -X utf8 -B "$controller/validate_six_view_scenes.py" recompose-atmosphere --out <实景试验目录>
```

默认使用 `36_6_6` 和 `53_7_35`。宽承托复验针对首轮因位置或接触淘汰的组合，沿用同一物品、场景尺度和接触门槛。雾场重合成按原签名读取物体与成对阴影 EXR，通过 alpha、遮挡、阴影和编辑范围检查后输出。旧视觉评分绑定旧图。程序证据和视觉复核尚未完整的图片保留为待审。

### 其余物品的局部几何精修

[部件细化配置](../tools/ps-model-controller/inventory-detail.example.json)按当前源模型散列锁定已知部件。铸件、机加工件、卷边和塑料盖使用各自的有限倒角宽度。检查修改部件的开边拓扑、整体外廓、重开后的网格、UV、贴图以及同机位完整轮廓。新模型保存为独立候选。

```powershell
$python = 'work/triposr-env/Scripts/python.exe'
$blender = 'work/blender45/blender-4.5.3-windows-x64/blender.exe'
$controller = 'tools/ps-model-controller'
& $python -X utf8 -B "$controller/report_inventory_details.py" --inventory work/detail-inventory.json
& $blender -b --factory-startup --python-exit-code 1 --python "$controller/refine_inventory_details.py" -- --inventory work/detail-inventory.json --recipe "$controller/inventory-detail.example.json" --output outputs/inventory-detail
& $python -X utf8 -B "$controller/report_inventory_details.py" --inventory work/detail-inventory.json --output outputs/inventory-detail
```

活动工作流中，05–07 和 38–39 为归档储罐。01、25 与 33 可继续查阅各自的专用细网格候选。这个入口处理其余 36 项。

## 准备与绑定

`prepare_reference_textures.py` 有两种输入。普通六视图提取用于审计局部样本及结构污染风险；`--flat-master` 用于已经检查过的无结构表面母图。不能因为六视图提取有输出文件就自动当作合格纹理。

本轮实测的直接拼片仍包含耳扣、铸字及重复纹样，因此未用于正式候选。内置 ImageGen 参考用户的阀门和灭火器风格生成两张平铺母图，提示词、原图散列和输出散列保存在项目的本地生成记录中。母图生成不是重新生成整个模型或现场图片。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/prepare_reference_textures.py --help

& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/diagnose_reference_materials.py --reference-manifest work/reference-materials-20261117/library.json --output outputs/reference-review --resolution 1024 --freeze-only

& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/diagnose_reference_materials.py --reference-manifest work/reference-materials-20261117/library.json --output outputs/reference-review --resolution 1024
```

库以当前 GLB 的 SHA-256 绑定条目，不按文件名数字猜身份。条目记录对应六视图、母图角色、材质估计及六通道散列。25 号蓝缸与 31 号黄缸不得混用身份；05–07 仍排除。来源仅为同类风格的图不能成为新模型身份依据。

## 材质实现

- 已确认的金属及漆面使用原材质分区。原色以区域统计保留，去掉原贴图中大面积受光后，结合母图的细节变化重新受光；保护区使用原着色器。
- 底色、粗糙度、锈蚀、擦伤、微高度、灰尘六图共享坐标。底色为 sRGB，其余为 Non-Color；粗糙度及微高度只是有界估计。亮暗不等于真实凹凸，估计微高度不超过 80 微米，不修改顶点位置。
- 三维物体坐标按尺寸估计换算纹样尺度，随机偏移绑定源资产；不读取背景颜色，不随背景重新染色。尺寸估计的来源写入结果，未核实的尺寸不能升级成现场尺度。
- 铁基氧化、露钢及非金属粉尘分别混合。未知合金不自动产生铁锈。浮灰上限 0.5，漆面半量。塑料、橡胶、标签、加工面和无法安全分离的区域保持原材质。
- 提供实际材质节点回读及 AOV，区分新表面响应和仅保留的源节点。换光导致高光变化、材质图不变是必要检查，不等于写实质量通过。

`effects.reference_material_manifest` 为显式可选字段，只支持 `surface_finish=physical_layers`。旧配置不设置此字段，保持历史语义。参考库、贴图、材质分区及执行器版本参与缓存校验；纯摄影参数不应使材质重新生成。

## 检查与边界

### 使用材质库做对照

对同一物品的裸钢、漆、橡胶及塑料分别接入材质包时，使用[分件 PBR 材质入口](component-pbr-materials.md)。

`build_blender_material_assets.py` 在 Blender 中将公开 PBR 贴图打包为可追加的材质资产。规格 JSON 为数组，每项包含 `name`、`provider`、`source_url` 以及 `channels` 下的 `base`、`roughness`、`metallic`、`normal`。将输出所在目录添加到 Blender 的 Asset Libraries。

`prepare_library_material_trial.py --spec <json> --baseline-library <json> --output <new-dir>` 将库贴图适配到已有的物品分区。规格另需 `id`、`assets`、`scale_m`、`oxide_value_limit` 及 `channels.height`，可用 `oxide_strength` 衰减估计锈蚀遮罩。每件可在 `only_materials` 中列出允许修改的源材质名。生成的 `library.json` 可直接用于完整模型诊断。独立比较换光下的粗糙度和高光后，再确定全局采用哪套材质。

独立入口按冻结清单输出原材质、新材质、换光、转向四张完整模型图。先行预检可以用 `--asset-ids 1 33 --variants source-left new-left --resolution 512`，但正式检查使用新目录。程序核对源散列、原/新轮廓、尺寸、候选文件重新打开及材质通道；缺少缓存时仅导出诊断几何，不启动重建、落稳或 AI。

没有可修改区域的资产必须显示“保留原貌／无有效修改”，不能计为成功重贴。诊断视角不证明真实支撑，模型图不施加现场薄雾、调色或相机模糊；用户确认模型效果后才进入场景测试。

代码检查、模型视觉、现场物理及最终合成分别记录。费用账本区分内置生图、交互评审、本地提取与 GPU 渲染；供应商未返回的 token、Credits 或金额记为未知。

原理参考：[Blender Principled BSDF](https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/principled.html)。纹理提供空间变化，金属、涂层、氧化及粉尘仍须由各自的表面响应解释。
