#!/usr/bin/env python3
import re


def main():
    # Open input and output handles
    infile = open("power.out", "r")
    out = open("power.csv", "w")

    # Helper to print to either stdout or your output file
    def write(line: str):
        out.write(line)

    num_re = re.compile(r"(\d+)")

    skip = 0
    total = 0

    for line in infile:
        if skip == 2:
            write(str(total) + "\n")
            skip = 0
            total = 0
        else:
            m = num_re.search(line)
            if m:
                total += int(m.group(1))
            skip += 1

    infile.close()
    if out:
        out.close()


if __name__ == "__main__":
    main()
