# 六视图参考材质

六视图用于学习磨损的外观和分布，不能直接作为校准后的 PBR 或六台相机的摄影测量。参考图中的白底、分格、文字、圆柱明暗和零件形状不能进入表面纹理。原模型、原贴图和生产默认保持不变，候选单独导出为打包依赖的 Blender 文件。

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
