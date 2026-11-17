# 模型写实诊断与定向执行

本轮先处理 03、22、31、33，并以 13 塑料胶罐作保护对照。先交付模型检查页，用户确认后再运行四件 × 三场景。`existing` 和原生产默认不变，候选不会替换源模型。

## 可复现入口

```powershell
# 仅本地模型诊断；不调用 AI，不生成场景图。
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/diagnose_realism_assets.py --refinement --resolution 1024 --add-turn-reference 33 --output outputs/asset-review-new

# 当前对话中单件、明确图片的 Token Plan 诊断；不是后台任务。
& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/interactive_token_plan.py --format asset_photographer --image <current.png> --image <source.png> --image <relight.png> --image <clay.png> --context-file <context.json> --question-file <question.txt> --out <new-review-directory>
```

诊断读取已有批次冻结姿态，不把旧现场接触状态升级为通过。无相关本地缓存时先按运行说明准备资产，不能用任意几何补齐。模型图统一输出完整画面；候选 `.blend` 打包依赖并重开验证。新增材质不改源 GLB。

现有 CLI/API 的 `effects` 可显式指定：

```json
{
  "surface_finish": "physical_layers",
  "surface_finish_manifest": "work/render-materials/nearby-worn-steel-v15/pack.json",
  "asset_refinement": "component_v1",
  "surface_dust": 0.5,
  "render_controls": {
    "oxide_gain": 1,
    "scratch_gain": 1,
    "roughness_shift": 0,
    "relief_gain": 1,
    "light_width_scale": 1,
    "fill_gain": 1
  }
}
```

这些控制是绝对值。氧化、擦伤与微凹凸倍率为 0.5/1/1.5；粗糙度偏移为 −0.08/0/0.08；光源尺寸与补光倍率为 0.75/1/1.25。它们只在对应模式、已知材质域和支持的诊断协议下开放。`fill_gain` 增大会降低主辅光比，不能与旧的主补光相对环境能量 `light_ratio` 混为一谈。材质变化不重做基础落稳，并复用无物体阴影；光照变化重算相关阴影。

## 诊断与执行合同

`photo-brief-v3` 保留七项评分，增加部件 ID、观察、假设、反证、根因、验证方法、参照角色和测量引用。旧报告仍可读。模型初次诊断、混合材质、小目标或无改善时启用详细模块，普通/详细视觉输出预算为 3072/6144；执行为 1024/2048。预算不是必须消耗量。旧交互 `photographer` 格式保留 v2，新的资产格式为 `asset_photographer`。

明确区分当前图、身份图、场景参照和受控诊断图。棚拍图只支持身份/结构，不决定现场雾量。宽面灯下的宽高光不是金属度错误的充分证据。源节点未改也不证明源材质写实，只排除“本轮修改了该节点”的归因。

每次执行绑定图片、诊断、状态、目标部件、单位和绝对参数。只选择注册操作；不输出自由 JSX/Python。材质与高光根因须对应实际材质/灯光读回证据；未知时输出 `needs_measurement`。局部问题不得退回整件修改。当前执行器每轮选择一项，满足最多三项的上限；多部件同名独立操作尚未开放。

单条语义矛盾可请求一次定向修复；有效兄弟发现保持，严重未知不能通过删除获得合格。网络/格式修复合计最多追加一次；记录精确 schema 错误但不写凭据或服务原始正文。修改后重新审查实际输出，重复参数、无有效像素变化或无改善时停止。

相对改善须提供 `reference_image_id` 和 `change_vs_reference`。完整同条件像素比较通过实际图片散列绑定；几乎不变只能说明没有可验证的变化，不证明质量通过。未解决发现的严重度不能被语义重试调低；执行前重验报告与状态。Photoshop 同时比较实际 RGB 像素，PNG 元数据改变不会触发额外复评。

Photoshop 独立入口、实际回读、部件蒙版和一次测量请求的合同见 [Photoshop 接口](photoshop-readback.md)。材质/几何缺陷不会自动变成曲线或浮灰操作。

## 资产与知识维护

部件档案按源模型散列绑定，不按背景配方分类。替换资产后旧档案不能自动沿用。允许有证据的逐件候选修复；分类不足的区域保留原材质，隐蔽结构不虚构。氧化、擦伤、沉积分别按物理尺度控制；加工面、聚合物和标识保护独立。浮灰上限 0.5，漆面半量，零值关闭新增层。

本轮增加源底色条带的受限清理，仅对确认的部件材质生成缓存变体；保留低频原色、细节和保护像素。它不改变原始贴图、粗糙度和法线。因此，粗糙度/法线自身的拉伸条纹仍需另行诊断，不能把底色清理当作全面修复。

项目技能为 [industrial-photo-review](../skills/industrial-photo-review/SKILL.md)。专业知识维护在模型输入的 material/scene 模块，技能说明证据及执行顺序；不得另写一套与运行时相反的默认参数。

## 外部工具取舍

- 提示词颗粒度参考 [DangerGoods_Generate 固定版本](https://github.com/LAWSSSS/DangerGoods_Generate/blob/f8a97e500f95112bc2c43840e49702c46cdfa8aa/generate_aliyun.py#L335)，不移植整图生成方案、固定尺寸或做旧百分比。
- [photo-editing-craft](https://github.com/aipiratesMunich/photo-editing-craft/tree/448f34fc8e53f406e8c002cf061c0471fe6030c2) 已固定在本地隔离目录，审阅工作流与合成章节后改写项目技能，附 MIT 许可。其手绘阴影、固定羽化、固定雾量不用于本项目验收。
- [drmedia/photoshop-mcp](https://github.com/drmedia/photoshop-mcp/tree/c9e4ed0ee98e02536c221a57b1f4967788cddecd) 与 [photoshop-mcp-imt](https://github.com/Immersive-Media-Technologies/photoshop-mcp-imt) 作为备选。现有 COM/MCP 在 Photoshop 26.6 已通过实际读写测试，因此本轮未换桥接器，未将候选 UXP 插件宣称为本机验证通过。

## 验证边界

软件、像素守卫、模型视觉、现场物理分别报告。第一阶段不提供场景融合合格率；局部阴影和灰底不能证明现场支撑。第二阶段仍须在用户确认模型后执行 12 组，目标 10/12 且每场景 3/4，未出图计入分母。费用没有可信价表或 Credits 返回时记未知，不按其他服务的价格计算。

离线报告入口 `report_asset_checkpoint.py` 接受 `--source-dir`、可重复 `--review-dir`、`--ps-dir`、`--out`，读取既有记录，不调用模型。完整结果与边界见 [本轮验证](history/DISPLAY_ASSET_CHECKPOINT.md)。
