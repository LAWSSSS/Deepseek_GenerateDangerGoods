# 实拍材质与阴天环境光 · 固定摆位验证

批量流程已恢复至 `aacef4a` 的渲染和摄影行为。低饱和点评沿用局部色度约束与扬尘估计。31 液压缸按原始基准 0.375，33 灭火器按 0.25，05–07 储罐继续排除。

[24 张回退原尺寸图](outputs/回退低饱和点评-20261111/index.html) · [8 组实拍材质对照](outputs/实拍材质环境光-点评-20261111/index.html) · [验证摘要](outputs/实拍材质环境光-点评-20261111/summary.json) · [成本账本](outputs/实拍材质环境光-点评-20261111/cost-ledger.json)

## 工具与素材

- Blender 4.5.3 LTS 的项目专用目录 `work/blender-addons/scripts/addons` 安装 Asset Importer 2.0.0。后台启用并核对两套素材的四类贴图、色彩空间和 2K 尺寸。保存并重新打开验证场景后渲染预览。批量渲染直接读取本地贴图缓存。 [插件源码固定版本](https://github.com/drune9d/blender-asset-importer/tree/7240b0d757c0fb83076e12d50559429cb5c85e62)
- Codex 用户目录已安装 using-blender 1.0.0。安装固定到 `9ebe57d3f45ce8a3dd735f00b438c35559d51aec`。 [技能源码固定版本](https://github.com/ellmos-ai/skills/blob/9ebe57d3f45ce8a3dd735f00b438c35559d51aec/skills/game-dev/using-blender/SKILL.md)
- 本地库缓存两套 2K 扫描贴图和 1K HDRI，共 26,741,264 字节原始素材。库清单记录下载地址、许可、校验值及派生图来源。粗糙度和法线只作用于原 UV 能确认的金属、漆面和锈面。贴图底色提取亮度细节，材料分区保留识别标记、聚合物与透明表面。粉尘为独立的非金属层。 [Rusty Metal 03](https://polyhaven.com/a/rusty_metal_03) · [Rusty Painted Metal](https://polyhaven.com/a/rusty_painted_metal) · [Overcast Soil](https://polyhaven.com/a/overcast_soil) · [CC0 许可](https://polyhaven.com/license)

## 验证结果

A 为回退低饱和图，B 为实拍材质，C 为实拍材质加局部校准阴天光与 Cycles 阴影捕捉。固定 13、22、31、33 与两张点评背景。程序复用相机、位置、尺度、遮挡与接触点。阴天光只按落点附近亮度和颜色估计，光向缺乏局部测量时沿水平方位平均 HDRI。

| 检查 | A 回退 | B 实拍材质 | C 材质与环境光 |
|---|---:|---:|---:|
| 完整原尺寸 PNG | 8/8 | 8/8 | 8/8 |
| 程序固定几何与像素检查 | 8/8 | 8/8 | 8/8 |
| 视觉摄影、接地与接缝达标 | 4/8 | 0/8 | 0/8 |

- 11 和 21 号盲评在一次格式重试后均无有效结果，保持待复核。六组有效评分中，B/C 仍有亮边、材质偏平、摄影颗粒不匹配和接触阴影偏弱。局部降色度通过数值检查，摄影效果仍低于门槛。24 项复验停在小样门槛，批量继续使用回退版。
- C 的八组均使用线性阴影捕捉通道。原始轮廓 IoU 最低 0.9915，超过 0.975。可见透明度、孔洞、前景遮挡与冻结基线一致，保护区、前景和范围外像素差异均为 0。44 个源模型散列一致。程序对隐藏接触和现场尺度的原有未知继续保留。
- 24 张回退图复用 20 张旧 PNG 和四张同尺寸旧材质对照。液压缸完整长轴 173/411 px，灭火器 177/271 px。回退复制与核验约 20.6 秒，0 次渲染和 API 请求。
- 104 项批量、44 项网页和模型检查通过。离线回放复现 16 张 B/C PNG 的文件散列及六组有效评审，耗时约 39.0 秒，0 次新增渲染和 API 请求。程序记录终止的格式错误，续跑复用该记录。

## 实际用量与复跑

小样生成与评审耗时 **123.9 秒**，其中 16 套物体渲染及配套的 32 次阴影相关渲染累计 **46.8 秒**。实际 **10 次 API 请求**，含两次格式重试，合计 **20,331 token**。按高低峰公布单价估算 **CNY 0.0233352–0.0466704**。本机电费与设备费用未定价。 [DeepSeek 官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

在项目根目录运行。首次准备前将上述固定版本的 `asset_importer.py` 放入项目专用目录。实际点评和离线回放各使用独立输出目录。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/feedback_rollback.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/prepare_display_pbr.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/validate_display_scanned.py --out 'outputs/实拍材质环境光-新批次'
```

追加 `--offline` 可使用已有场景、评审和渲染缓存复核。达到 7/8 小样门槛才进入 24 项复验。达到 20/24 后等待用户整体风格确认。
