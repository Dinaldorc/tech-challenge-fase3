"""
Divide um CSV grande demais para o GitHub (>100 MB) em partes menores,
repetindo o cabeçalho em cada parte, e permite reconstituir o arquivo
original a partir das partes.

Uso:
    python scripts/split_csv.py split <arquivo.csv> [--max-mb 90]
    python scripts/split_csv.py join <arquivo.csv>

O comando `split` gera <arquivo>.part1.csv, <arquivo>.part2.csv, ... no
mesmo diretório e apaga o arquivo original. O comando `join` faz o
inverso: lê as partes e recria <arquivo.csv>.
"""
from __future__ import annotations
import argparse
import glob
import os

ENCODING = "ISO-8859-1"


def split(path: str, max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    base, ext = os.path.splitext(path)

    with open(path, "r", encoding=ENCODING, newline="") as f:
        header = f.readline()

        part_num = 1
        out = open(f"{base}.part{part_num}{ext}", "w", encoding=ENCODING, newline="")
        out.write(header)
        size = len(header.encode(ENCODING))

        for line in f:
            line_size = len(line.encode(ENCODING))
            if size + line_size > max_bytes:
                out.close()
                part_num += 1
                out = open(f"{base}.part{part_num}{ext}", "w", encoding=ENCODING, newline="")
                out.write(header)
                size = len(header.encode(ENCODING))
            out.write(line)
            size += line_size
        out.close()

    os.remove(path)
    print(f"'{path}' dividido em {part_num} parte(s).")


def join(path: str) -> None:
    base, ext = os.path.splitext(path)
    parts = sorted(glob.glob(f"{base}.part*{ext}"))
    if not parts:
        raise FileNotFoundError(f"Nenhuma parte encontrada para '{path}'")

    with open(path, "w", encoding=ENCODING, newline="") as out:
        header = None
        for i, part in enumerate(parts):
            with open(part, "r", encoding=ENCODING, newline="") as f:
                first_line = f.readline()
                if i == 0:
                    header = first_line
                    out.write(header)
                for line in f:
                    out.write(line)
    print(f"'{path}' reconstituído a partir de {len(parts)} parte(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Divide o CSV em partes menores")
    p_split.add_argument("path")
    p_split.add_argument("--max-mb", type=int, default=90)

    p_join = sub.add_parser("join", help="Reconstitui o CSV a partir das partes")
    p_join.add_argument("path")

    args = parser.parse_args()
    if args.command == "split":
        split(args.path, args.max_mb)
    else:
        join(args.path)
