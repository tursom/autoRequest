#!/usr/bin/env python
import os
import sys


def main():
    remove_head = 0
    remove_end = 0
    rename_files = []
    i = iter(sys.argv[1:])
    for arg in i:
        if arg == "--head":
            remove_head = int(next(i))
        elif arg == "--end":
            remove_end = int(next(i))
        else:
            rename_files.append(arg)
    for f in rename_files:
        print(f)
        for f_old in open(f, "r").readlines():
            f_old = f_old[:-1]
            if remove_end != 0:
                f_new = f_old[remove_head:-remove_end]
            else:
                f_new = f_old[remove_head:]
            print(f_old)
            print(f_new)
            os.renames(f_old, f_new)


if __name__ == '__main__':
    main()
