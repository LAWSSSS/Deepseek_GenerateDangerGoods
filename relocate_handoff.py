"""Resolve portable placeholders after extracting this project on Windows."""
import argparse
import json
from pathlib import Path
from urllib.parse import quote

TOKENS=("__HANDOFF_PROJECT_URI__","__HANDOFF_PROJECT__")
TEXT_EXTENSIONS={".json",".jsonl",".csv",".html",".md",".txt",".log",".py",".ps1",".jsx",".yaml",".yml",".js",".mjs",".cjs",".toml",".ini",".cfg",".xml",".sh",".tsx",".ts",".bat",".cmd",".svg"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    root=args.project.resolve()
    state=root/"work"/"relocation-state.json"
    old=json.loads(state.read_text(encoding="utf8"))["project"] if state.exists() else None
    current=root.as_posix()
    replacements=[(TOKENS[0],quote(current,safe="/:")),(TOKENS[1],current)]
    if old and old!=current:
        replacements.extend([(quote(old,safe="/:"),quote(current,safe="/:")),(old,current)])
    changed=0
    for directory in (root/"outputs",root/"external",root/"work"/"v2-cache"):
        if not directory.exists():
            continue
        for file in directory.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            data=file.read_bytes()
            encoding="utf-16" if data.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig"
            try:
                original=data.decode(encoding)
            except UnicodeDecodeError:
                encoding="gb18030"
                original=data.decode(encoding)
            updated=original
            for source,target in replacements:
                updated=updated.replace(source,target)
            if updated!=original:
                if encoding=="utf-8-sig":
                    encoded=(b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b"")+updated.encode("utf8")
                else:
                    encoded=updated.encode(encoding)
                temporary=file.with_name(file.name+".relocating")
                temporary.write_bytes(encoded)
                temporary.replace(file)
                changed+=1
    state.parent.mkdir(parents=True,exist_ok=True)
    state.write_text(json.dumps({"project":current},ensure_ascii=False,indent=2),encoding="utf8")
    print(f"Relocated {changed} metadata files to {current}")


if __name__=="__main__":
    main()
