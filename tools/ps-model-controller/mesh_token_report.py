"""Summarize per-image API usage across explicitly listed pilot versions."""
import argparse
import csv
import html
from pathlib import Path


def integer(value):
    return int(value) if value not in (None,"") else None


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("versions",type=Path,nargs="+")
    args=cli.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for version in args.versions:
        with (version/"token-usage.csv").open(encoding="utf-8-sig",newline="") as file:
            rows=list(csv.DictReader(file))
        for row in rows:
            entries.append({"version":version.name,"number":int(row["number"]),
                            "api_calls":integer(row["api_calls"]),"total_tokens":integer(row["total_tokens"]),
                            "prompt_tokens":integer(row["prompt_tokens"]),
                            "completion_tokens":integer(row["completion_tokens"]),
                            "cache_hit_tokens":integer(row["prompt_cache_hit_tokens"]),
                            "unreported_calls":integer(row["unreported_calls"])})
    pending=args.out/"versions.tmp"
    fields=["version","number","api_calls","total_tokens","prompt_tokens","completion_tokens",
            "cache_hit_tokens","unreported_calls"]
    with pending.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    pending.replace(args.out/"各版本逐图token.csv")
    lines=["<!doctype html><meta charset='utf-8'><title>三维试验 token</title>",
           "<style>body{font-family:Arial,sans-serif;margin:2em}td,th{padding:.5em;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>",
           "<h1>各版本 DeepSeek token</h1><p>0 表示本版本没有实际请求，可能复用了先前缓存；未知用量不会记为 0。共享分析计入实际发起请求的图片。</p>",
           "<p><a href='各版本逐图token.csv'>下载明细 CSV</a></p><table><tr><th>版本</th><th>03</th><th>10</th><th>27</th><th>30</th><th>合计</th></tr>"]
    for version in args.versions:
        selected=[row for row in entries if row["version"]==version.name]
        values={row["number"]:row["total_tokens"] for row in selected}
        total=sum(values.values()) if len(values)==4 and all(x is not None for x in values.values()) else None
        columns=[values.get(n) for n in (3,10,27,30)]+[total]
        lines.append("<tr><td>"+html.escape(version.name)+"</td>"+
                     "".join("<td>"+(str(value) if value is not None else "未知")+"</td>" for value in columns)+"</tr>")
    lines.append("</table>")
    pending=args.out/"index.tmp"
    pending.write_text("".join(lines),encoding="utf8")
    pending.replace(args.out/"index.html")
    print("Wrote",len(entries),"per-image version records")


if __name__=="__main__":
    main()
