# 分件 PBR 材质

`component-pbr-v1` 将材质包绑定到源 GLB 散列、对象名和材质槽。试验覆盖 01 DN350 阀门、25 蓝色液压缸、33 灭火器。使用原网格、UV 和法线，候选 `.blend` 内嵌本地贴图。现场融合需用相同摆位另行验证。

## 教程在项目中的用法

- **分别表达漆、裸钢和锈。** 完整漆和铁氧化使用介电表面，露钢使用金属反射。黄铜单独使用带色金属反射。底色按 sRGB 读入，金属度、粗糙度和高度按数据通道处理。 [Adobe 金属度与粗糙度教程](https://www.adobe.com/us/learn/substance-3d-designer/web/the-pbr-guide-part-2) · [Blender Principled BSDF](https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/principled.html)
- **让磨损的颜色和起伏对得上。** 一份掉漆遮罩同时控制露钢和薄漆边缘的微凹凸。氧化独立分布，在露钢和凹处有较高权重。印刷字从源 UV 取遮罩保留，阀门的独立标字对象沿用源材质。 [Adobe 遮罩使用说明](https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/masking-and-effects) · [Substance Anchor Point 教程](https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/anchor-point)
- **按表面尺度投射。** 物体空间三向投射让纹样跟随零件，按照物品尺寸估计实际纹理周期。标量高度驱动受限微凹凸，橡胶和塑料可叠加源 UV 法线。预处理减弱贴图中的大范围明暗，保留各通道的中小尺度变化。 [Blender 图像贴图节点](https://docs.blender.org/manual/en/4.5/render/shader_nodes/textures/image.html) · [Blender Bump 节点](https://docs.blender.org/manual/en/4.5/render/shader_nodes/vector/bump.html)
- **换光复验。** 同机位改变宽面光源，读取磨损通道与高光分量，再观察转向画面。颜色锚点、投射周期、粗糙度范围和微凹凸厚度是按原貌调出的候选参数，现场材质及实际尺寸仍需对应证据。

## 已用材质包

| 部件 | 材质包和取用范围 |
| --- | --- |
| 阀体、钢制接头 | [ambientCG Metal056A](https://ambientcg.com/view?id=Metal056A) 的钢表面细纹与粗糙度；已确认铁质区域另加 [Poly Haven Rusty Metal 02](https://polyhaven.com/a/rusty_metal_02) 的棕色氧化层 |
| 加工杆、套圈与露钢区 | [ambientCG Metal010](https://ambientcg.com/view?id=Metal010) 按加工面和露钢面分别校准粗糙度与纹理尺度 |
| 蓝色手轮、蓝缸筒身与耳座 | [ambientCG PaintedMetal002](https://ambientcg.com/view?id=PaintedMetal002) 的粗糙度和露钢遮罩，蓝色以原部件为依据 |
| 灭火器筒身和压把 | [ambientCG PaintedMetal004](https://ambientcg.com/view?id=PaintedMetal004) 的划痕、露钢和粗糙度，保留红漆与印字 |
| 密封圈、软管 | [ambientCG Rubber004](https://ambientcg.com/view?id=Rubber004) 是地垫材质，本轮取细颗粒和粗糙度，软管的肋纹仍来自原模型 |
| 底座、喷嘴 | [ambientCG Plastic006](https://ambientcg.com/view?id=Plastic006) 的塑料细粒，保留原成型纹理 |
| 阀件、导向环 | [TextureCan Worn Yellow Brass](https://www.texturecan.com/details/497/) 的黄铜底色、粗糙度与受限微凹凸 |

涉用贴图为 CC0。材质清单记录来源、原通道及派生图的 SHA-256。 [ambientCG 许可](https://docs.ambientcg.com/license/) · [Poly Haven 许可](https://polyhaven.com/license/) · [TextureCan 许可](https://www.texturecan.com/terms/)

## 复现或增加模型

1. 按 [试验配置](../tools/ps-model-controller/component-pbr.example.json) 中的 `source` 获取贴图，将各通道放在 `channels` 对应的路径。`--source-root` 指定这些相对路径的起点，省略时以配置目录为起点。`assets` 使用仓库图册 ID，01 写作 `1`。
2. 对新资产先核对实物照片、源部件名和材质槽。确认可使用独立材质的零件后填写 `parts`、`material`、`package`。已确认是漆面钢基底时可设置 `substrate` 和 `oxide`。未列出的部件保留源 PBR。标字保护来自现有 UV，增加新型印刷时应先核对遮罩。
3. 在仓库根目录运行。输出目录应为新目录，已有缓存可供程序复用。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/prepare_component_pbr.py --spec tools/ps-model-controller/component-pbr.example.json --source-root work/component-pbr-trial --output work/component-pbr-reproduction
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/diagnose_reference_materials.py --reference-manifest work/component-pbr-reproduction/library.json --output outputs/component-pbr-reproduction --resolution 1024 --device OPTIX --pose-mode inspection --surface-dust 0.24
```

`--freeze-only` 可先验证物品与贴图的绑定。完整运行生成源材质、新材质、换光、转向四张全画幅及候选 `.blend`。程序对照 alpha，回读材质节点和贴图散列，并核对重开后的网格、UV 与角法线。批量侧显式传 `effects.surface_finish=physical_layers` 和 `effects.reference_material_manifest` 可选择该库。默认继续使用生产配置。
