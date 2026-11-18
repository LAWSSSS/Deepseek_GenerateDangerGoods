"""Evidence-bound asset descriptors; recipes use material classes, never backgrounds.

These declarations supplement the existing authored UV partitions. They do not
repaint unknown texels, infer a precise product model, or authorize geometry edits.
"""
import copy
from .common import signature

VERSION = 'asset-material-evidence-v1'
PROFILES = {
 '5e779052a25e4ec035857ab66031c9feddc62daf0d49873bb4eed2cf2d4b8e47': {
  'asset_id':'13','name':'胶罐保护对照','basis':'confirmed single plastic container; named source polymers',
  'parts':[{'id':'container','material':'plastic','evidence':'白色塑料罐体与盖','protect':['全部源材质','单只数量'],'allow':[]}],
  'material_evidence':{},
 },
 'a66b84f411cf86a3719225aebed54e59e59b854f1062efd9171282f1486da327': {
  'asset_id':'3', 'name':'轴承座', 'basis':'P210 identity photograph, six-view rebuild and named source materials',
  'parts':[
   {'id':'housing','material':'painted_cast_iron','evidence':'蓝色铸铁座体、安装底脚','protect':['孔沿','铸字','安装槽'],'allow':['局部露铁','片状氧化','不均匀擦伤']},
   {'id':'bearing','material':'machined_steel','evidence':'内外轴承圈','protect':['加工面','内孔','紧定螺钉'],'allow':[]},
   {'id':'seal','material':'rubber','evidence':'黑色密封环','protect':['全部源材质'],'allow':[]},
   {'id':'zerk','material':'chrome','evidence':'顶部油嘴','protect':['加工反射与几何'],'allow':[]}],
  'material_evidence':{'03 blue paint':{'metal_family':'ferrous','part_id':'housing'}, '03 dark steel':{'metal_family':'ferrous','part_id':'housing'}},
  'source_detail_cleanup':{'strength':.75,'neutral_only':False,'evidence':'source PBR housing has medium-scale directional colour bands absent from the cast-grain identity photograph'},
 },
 '3396c3c6ec2f3f3453d1e3391964819891fb49372f566539d29dccd73c025568': {
  'asset_id':'22','name':'气缸','basis':'source photograph and authored aluminium/steel/brass atlas tiles',
  'parts':[
   {'id':'barrel_caps','material':'nonferrous','evidence':'银色缸筒、方形端盖；源图集标注铝','protect':['曲面层次','孔口'],'allow':['浅氧化','不规则擦痕','槽内积尘']},
   {'id':'tie_rods','material':'steel','evidence':'四根拉杆，图集独立钢区域','protect':['杆件轮廓'],'allow':['局部锈点','划痕']},
   {'id':'piston','material':'machined_metal','evidence':'镀层活塞杆','protect':['镀层','螺纹','螺母'],'allow':[]},
   {'id':'print_seals','material':'protected','evidence':'绿色气口标记及未知图集块','protect':['全部源材质'],'allow':[]}],
  'material_evidence':{},
  'source_detail_cleanup':{'strength':.75,'neutral_only':True,'evidence':'source PBR has broad parallel streaks across barrel and cast endcaps; identity photo supports fine metal grain, not wooden bands'},
 },
 'c4cb138c29bedf2d2bc24e776622f38e238d176064f2e577236548bfe2b31c0b': {
  'asset_id':'31','name':'液压缸','basis':'confirmed closed barrel revision; authored paint/steel/protected atlas tiles',
  'parts':[
   {'id':'barrel','material':'painted_steel','evidence':'黄色封闭缸筒','protect':['封闭筒壁','原黄色类别'],'allow':['长短不一擦伤','局部重锈','稀疏积尘']},
   {'id':'eyes_ports','material':'steel','evidence':'两端耳环及油口','protect':['孔洞与连接'],'allow':['局部氧化','凹槽积垢']},
   {'id':'piston','material':'chrome','evidence':'伸出杆，原图集保护块','protect':['加工反射','杆面'],'allow':[]},
   {'id':'seal','material':'rubber','evidence':'端盖密封环，原图集保护块','protect':['全部源材质'],'allow':[]}],
  'material_evidence':{},
 },
 '5005f43274d9edfe0d99ca9ab8cdba809b914b3c2a0b89988c425a6fadfc5f2f': {
  'asset_id':'33','name':'灭火器','basis':'source red painted body and label UV rectangle; separated valve/hose materials',
  'parts':[
   {'id':'body_lever','material':'painted_steel','evidence':'红漆筒体与压把','protect':['红色类别','标签'],'allow':['局部露钢','锈点','稀疏长划痕']},
   {'id':'valve','material':'brass','evidence':'独立黄铜阀体材质','protect':['黄铜材质'],'allow':['浅擦伤','局部积尘']},
   {'id':'hose_base','material':'polymer','evidence':'黑色软管与底座','protect':['全部源材质'],'allow':[]},
   {'id':'label','material':'print','evidence':'已记录保护 UV 区域','protect':['文字与边界'],'allow':[]}],
  'material_evidence':{'Red painted lever with chipped edges, PBR':{'metal_family':'ferrous','part_id':'body_lever'}},
 },
}


def load_profile(mesh_hash):
    value=PROFILES.get(mesh_hash)
    if value is None:
        return None
    profile=copy.deepcopy(value)
    profile.update(version=VERSION,source_mesh_hash=mesh_hash,
                   geometry_policy='candidate_only_with_visible_reference_evidence',
                   appearance_policy='asset_stable_across_backgrounds')
    profile['signature']=signature(profile)
    return profile


def descriptor(profile):
    return {k:copy.deepcopy(profile[k]) for k in ('version','asset_id','name','basis','parts','signature')}
