"""Resolve photographed assets and their dimension evidence by ID or content."""
from pathlib import Path

from pydantic import Field, model_validator

from metric_scale import ObjectSize, Point, ScaleReview
from run import Strict, load
from scene_v2 import digest


class AssetFeature(Strict):
    label: str = Field(min_length=1,max_length=40)
    point: Point


class AssetFile(Strict):
    file: str = Field(min_length=1,max_length=600)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    features: list[AssetFeature] = Field(default_factory=list,max_length=3)
    prepared_generated: bool = Field(default=False,exclude_if=lambda value:not value)


class AssetRecord(AssetFile):
    asset_id: str = Field(pattern=r"^asset-[a-f0-9]{12}$")
    category: str = Field(min_length=1,max_length=80)
    model: str | None = None
    object_size: ObjectSize | None = None
    evidence: list[str] = Field(default_factory=list)
    note: str = ""
    derivatives: list[AssetFile] = Field(default_factory=list)


class Inventory(Strict):
    schema_version: int = Field(ge=1,le=1)
    source_root: str
    assets: list[AssetRecord]

    @model_validator(mode="after")
    def unique_keys(self):
        ids = [asset.asset_id for asset in self.assets]
        hashes = [item.sha256 for asset in self.assets for item in (asset,*asset.derivatives)]
        if len(set(ids))!=len(ids) or len(set(hashes))!=len(hashes):
            raise ValueError("duplicate asset ID or content hash")
        return self


class Catalog:
    def __init__(self,path,asset_root=None):
        inventory = load(Inventory,path)
        self.root = Path(asset_root) if asset_root else path.parent/inventory.source_root
        self.by_id = {record.asset_id:record for record in inventory.assets}
        self.by_hash = {item.sha256:record for record in inventory.assets
                        for item in (record,*record.derivatives)}

    def resolve(self,value,base):
        record = self.by_id.get(value)
        if record is not None:
            source = (self.root/record.file).resolve(strict=True)
            sha = digest(source)
            if sha!=record.sha256:
                raise ScaleReview("catalog_asset_changed","素材 ID 对应文件与入库哈希不符")
            return source,record,sha
        if value.startswith("asset-") and not (base/value).is_file():
            raise ScaleReview("unknown_asset","素材 ID 不在物品库中")
        source = (base/value).resolve(strict=True)
        sha = digest(source)
        return source,self.by_hash.get(sha),sha
