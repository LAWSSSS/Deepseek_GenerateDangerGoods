# 生成贴图同摆位试验

使用内置 ImageGen 各生成一张裸钢氧化、旧漆剥落和灰尘团聚贴图，三张均为 1254×1254。原始图、处理后的数据图、提示词、尺寸和 SHA-256 保存在本机 [材质清单](work/render-materials/generated-pbr-v1/pack.json)。数值图对接缝做连续化并分别提取氧化、细坑、掉漆、划痕、粉尘覆盖及颗粒。三张成品数据图首尾边缘差均为 0，半周期相关性最高 0.095。

Blender 按归一化后的统一物体空间尺度投射贴图。氧化和掉漆局部改变底色与反射，独立的灰尘层在朝上和凹陷处增加覆盖。保留源法线并叠加浅凹凸。22、31 采用与源 GLB 散列绑定的图集分区。镀铬、轴承加工面、印刷标记、密封件和未知区域保持源材质。浮灰上限 0.5，漆面半量，0 关闭浮灰层。 [Blender 凹凸节点](https://docs.blender.org/manual/en/4.5/render/shader_nodes/vector/bump.html)

## 试验结果

液压缸 `36_6_6` 首先用三个 448 px 预览检查受光和纹理。光向从左移到右时，材质分布均差小于 0.000001，线性高光均差 0.0061。原光向下转动模型 55° 后，共同可见区域的材质分布发生变化。首张完整对照复现上一版原图后再生成新版，长轴保持 376 px。另完成六组 03、22、31 × `36_6_24`、`53_7_35`。

七张新版均为 3840×2160。固定 alpha、孔洞、前景、保护区及编辑范围外像素检查通过，接触锚点变化为 0 px，原始渲染轮廓 IoU 为 0.9995–1.0000。七项长轴、薄雾和相机模糊核对一致。44 个源 GLB 散列一致。

| 场景 | 物品 | 盲评偏好 | 新版材质 / 纹理 | 判定 |
|---|---|---|---:|---|
| 36_6_24 | 03 轴承座 | 新 | 3 / 3 | 有遮挡疑点，诊断 |
| 36_6_24 | 22 气缸 | 旧 | 3 / 3 | 原版反光和细节更清楚 |
| 36_6_24 | 31 液压缸 | 新 | 3 / 3 | 摄影小样达标，现场证据待审 |
| 53_7_35 | 03 轴承座 | 旧 | 2 / 2 | 有姿态疑点，诊断 |
| 53_7_35 | 22 气缸 | 新 | 3 / 3 | 摄影小样达标，现场证据待审 |
| 53_7_35 | 31 液压缸 | 相近 | 2 / 2 | 全景磨损、粉尘与反光仍不足 |

三组偏好新版且材质、纹理均达到 3 分。计入程序与视觉缺陷检查后为 2/6，低于 5/6 门槛。两个轴承座的遮挡和姿态提示来自盲评，模型证据的文字也有歧义。现场承托与尺寸仍为估计，轴承座两张供诊断，另外五张为待审。首张液压缸为固定几何及光学对照，六组扩测参与盲评。

[原尺寸新旧完整画面与材质分布](outputs/生成贴图-固定摆位-20261113-v2/index.html) · [逐项验证与用量](outputs/生成贴图-固定摆位-20261113-v2/validation.json)

## 运行与用量

每个正式版本渲染物体和有物体阴影各一次，七个无物体阴影均复用。七组小样合计 78.9 秒，Blender worker 占 48.2 秒，包含三张光向和转向预览。早期预览已写出一张图，因 Windows 中文路径下 EXR 读取失败增加 2.8 秒 Blender 时间，改用字节解码后完成。

内置 ImageGen 使用 3 次，实际金额未返回。摄影盲评 6 组，DeepSeek 实际发出 8 次请求，其中格式重试 2 次，合计 30,073 token，估算 ¥0.0427–0.0853。首次评审回包已缓存，修复报告绑定后直接复用。图像生成和本机电费无可核算金额。 [DeepSeek 计价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

批量使用 `existing` 默认。试用新材质可在已有效的批次配置里增加以下字段。旧锈层开关应使用默认零值。

```json
{
  "effects": {
    "surface_finish": "generated_pbr",
    "surface_finish_manifest": "work/render-materials/generated-pbr-v1/pack.json",
    "surface_dust": 0.5,
    "weathering": "preserve_source",
    "fog_mode": "local_auto",
    "fog_extra": 0,
    "pool_kernel": 1,
    "lighting": "local",
    "shadow_mode": "matched_receiver"
  }
}
```

固定小样复跑需保留本机上一版的两组历史目录、场景缓存和模型库。使用新输出目录可重跑，`--offline` 跳过在线盲评。`report` 从同一目录重建展示页。

```powershell
$python = 'work/triposr-env/Scripts/python.exe'
$controller = 'outputs/ps-model-controller'
& $python -X utf8 -B "$controller/validate_generated_pbr.py" all --offline --out 'outputs/生成贴图-复测'
& $python -X utf8 -B "$controller/validate_generated_pbr.py" review --out 'outputs/生成贴图-复测'
```

贴图内容签名绑定物体和有物体阴影，与无物体基准分开。源素材及 PNG 保存在本机，Git 收录代码、提示词、图集分区与记录。99 项相关回归通过，包含真实 Blender 材质节点、输入限幅、遮罩、散列和缓存检查。
