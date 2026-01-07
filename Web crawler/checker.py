#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, re, csv
from collections import OrderedDict

def iter_names_from_structure(obj):
    """遞迴走訪任何 Python 結構，擷取 key == 'name' 的值"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "name" and isinstance(v, (str, int, float)):
                yield str(v).strip()
            else:
                yield from iter_names_from_structure(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_names_from_structure(item)

def extract_names_yaml_first(text):
    """優先用 PyYAML；失敗則改用正則"""
    names = []
    try:
        import yaml  # PyYAML
        data = yaml.safe_load(text)
        if data is not None:
            names = list(iter_names_from_structure(data))
    except Exception:
        pass  # 掉回正則

    if not names:
        # 寬鬆正則：抓行首/縮排後的 name: 值，忽略註解
        pattern = re.compile(r'^\s*name\s*:\s*([^\n#]+)', re.MULTILINE)
        names = [m.group(1).strip() for m in pattern.finditer(text)]

    # 去重且保序
    seen = OrderedDict()
    for n in names:
        if n != "":
            seen.setdefault(n, None)
    return list(seen.keys())

def main():
    # 用法：
    #   python extract_names.py config.yaml
    # 或把內容管線進來：
    #   cat config.yaml | python extract_names.py
    if not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        if len(sys.argv) < 2:
            print("用法: python extract_names.py <config.yaml>（或使用管線）")
            sys.exit(1)
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            content = f.read()

    names = extract_names_yaml_first(content)
    if not names:
        print("未找到任何 name:")
        sys.exit(2)

    # 輸出 CSV
    out_path = "names_bul.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for n in names:
            writer.writerow([n])

    # 同時在終端顯示摘要
    print(f"共找到 {len(names)} 個 name，已輸出至 {out_path}:")
    for n in names:
        print("-", n)

if __name__ == "__main__":
    main()
