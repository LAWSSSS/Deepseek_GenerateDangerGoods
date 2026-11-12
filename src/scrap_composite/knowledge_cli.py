"""Offline history import and evidence report: python -m scrap_composite.knowledge_cli."""
import argparse
import json
from pathlib import Path

from .knowledge import KnowledgeBase, recommend


def discover_history(root):
    """Whitelist public result artifacts; configuration and credential files excluded."""
    root=Path(root)
    if not root.is_dir():
        raise ValueError("history root is not a directory")
    paths=set()
    for pattern in ("results.json", "results.jsonl", "analysis.json", "cost-ledger.json", "usage-*.json"):
        paths.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(paths)


def main(argv=None):
    parser=argparse.ArgumentParser(description="离线参数证据库；不调用 API、渲染或物理模拟")
    parser.add_argument("--db",default="work/knowledge/history.sqlite")
    commands=parser.add_subparsers(dest="command",required=True)
    ingest=commands.add_parser("import",help="导入指定历史结果和请求账本")
    ingest.add_argument("paths",nargs="*")
    ingest.add_argument("--history-root",action="append",default=[])
    commands.add_parser("derive",help="归纳53历史对照规则，用36配对数据独立验证")
    report=commands.add_parser("report",help="导出含完整图片链接的本地报告")
    report.add_argument("--out",default="work/knowledge/report")
    proposal=commands.add_parser("recommend",help="只读建议；无匹配证据时不修改参数")
    proposal.add_argument("context",help="JSON containing features and versions")
    args=parser.parse_args(argv)
    if args.command=="recommend":
        context=json.loads(Path(args.context).read_text(encoding="utf8"))
        result=recommend(context["features"],context["versions"],db_path=args.db)
    else:
        with KnowledgeBase(args.db) as knowledge:
            if args.command=="import":
                paths=[Path(p) for p in args.paths]
                for root in args.history_root:paths.extend(discover_history(root))
                if not paths:parser.error("import needs paths or --history-root")
                result=knowledge.import_history(sorted(set(paths)))
            elif args.command=="derive":
                result={"comparisons":knowledge.derive_trials(),"rules":knowledge.induce_rules()}
            else:result=knowledge.export_report(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
