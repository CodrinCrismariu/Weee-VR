"""Copy one architecture slice out of a universal (fat) Mach-O file, like `lipo -thin`.

usage: python3 macho-slice.py <fat-input> <arch> <output>     (arch: arm64)
"""
import struct
import sys

CPU_TYPES = {"arm64": 0x0100000C}


def main(src, arch, dst):
    with open(src, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    if magic not in (0xCAFEBABE, 0xCAFEBABF):
        sys.exit(f"{src} is not a universal binary")
    wide = magic == 0xCAFEBABF
    count = struct.unpack(">I", data[4:8])[0]
    entry, fmt = (32, ">iiQQII") if wide else (20, ">iiIII")
    for i in range(count):
        cputype, _, offset, size = struct.unpack(fmt, data[8 + i * entry: 8 + (i + 1) * entry])[:4]
        if cputype == CPU_TYPES[arch]:
            with open(dst, "wb") as out:
                out.write(data[offset:offset + size])
            print(f"{arch}: {size / 1e6:.1f} MB -> {dst}")
            return
    sys.exit(f"no {arch} slice in {src}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
