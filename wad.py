"""Wii WAD files (WiiWare, Virtual Console, channels): contents, decryption, the executable.

    python -m wiikit.wad TITLE.wad --info
    python -m wiikit.wad TITLE.wad --extract out/

A WAD is an installable title: a header, the certificate chain, the ticket,
the TMD and the title's contents, each section 64-byte aligned.

Header (big-endian): u32 header size (0x20), u16 type ('Is' or 'ib'), u16
      version, u32 certificate chain size, u32 reserved, u32 ticket size,
      u32 TMD size, u32 data size, u32 footer size.
Ticket
      The encrypted title key at 0x1BF, the title id at 0x1DC (IV = title id
      + 8 zero bytes), the common-key index at 0x1F1 (disc.COMMON_KEYS).
TMD   The IOS at 0x184 (u64), the title id at 0x18C, the group (maker) id at
      0x198, the region at 0x19C, the title version at 0x1DC, the content
      count at 0x1DE, the boot content's index at 0x1E0, then 36-byte content
      records from 0x1E4: u32 id, u16 index, u16 type, u64 size, SHA-1.
Contents
      AES-128-CBC with the title key, IV = the content index (u16) + 14 zero
      bytes, each padded to 16 bytes; the SHA-1 is of the decrypted content.

The boot content is often not the game: WiiWare boots Nintendo's NAND loader,
a small DOL that loads the game's executable from another content, as a DOL
or compressed with LZ77 (types 0x10 and 0x11). --extract writes the tree
wiiboot runs:

    tmd.bin  ticket.bin  cert.bin
    content/<id>.app     every content, decrypted and checked
    sys/main.dol         the game's executable: the DOL the loader loads (the
                         one DOL among the other contents, decompressed), or
                         the boot content itself when it is the game
"""
import argparse
import hashlib
import os
import struct

from . import aes
from .disc import COMMON_KEYS


def _align(v, a=0x40):
    return (v + a - 1) & ~(a - 1)


def _be32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def lz77_decompress(data):
    """Nintendo's LZ77: type 0x10 (DS/GBA) and 0x11 (LZ11), little-endian size.
    None if `data` is neither."""
    if len(data) < 4 or data[0] not in (0x10, 0x11):
        return None
    size, pos = int.from_bytes(data[1:4], "little"), 4
    if size == 0:
        size, pos = int.from_bytes(data[4:8], "little"), 8
    out = bytearray()
    try:
        while len(out) < size:
            flags = data[pos]
            pos += 1
            for bit in range(7, -1, -1):
                if len(out) >= size:
                    break
                if not flags & (1 << bit):
                    out.append(data[pos])
                    pos += 1
                    continue
                b1 = data[pos]
                if data[0] == 0x10:
                    b2 = data[pos + 1]
                    length, disp, pos = (b1 >> 4) + 3, (((b1 & 0xF) << 8) | b2) + 1, pos + 2
                elif b1 >> 4 == 0:
                    b2, b3 = data[pos + 1], data[pos + 2]
                    length = (((b1 & 0xF) << 4) | (b2 >> 4)) + 0x11
                    disp, pos = (((b2 & 0xF) << 8) | b3) + 1, pos + 3
                elif b1 >> 4 == 1:
                    b2, b3, b4 = data[pos + 1], data[pos + 2], data[pos + 3]
                    length = (((b1 & 0xF) << 12) | (b2 << 4) | (b3 >> 4)) + 0x111
                    disp, pos = (((b3 & 0xF) << 8) | b4) + 1, pos + 4
                else:
                    b2 = data[pos + 1]
                    length, disp, pos = (b1 >> 4) + 1, (((b1 & 0xF) << 8) | b2) + 1, pos + 2
                start = len(out) - disp
                if start < 0:
                    return None
                for i in range(length):
                    out.append(out[start + i])
    except IndexError:
        return None
    return bytes(out)


def is_dol(data):
    """A DOL: text section 0 right after the 0x100-byte header, sections and
    entry point in MEM1 (virtual or physical)."""
    if len(data) < 0x100 or _be32(data, 0) != 0x100:
        return False
    text0, entry = _be32(data, 0x48), _be32(data, 0xE0)
    return (text0 & 0x3FFFFFFF) < 0x01800000 and (entry & 0x3FFFFFFF) < 0x01800000


class Wad:
    def __init__(self, path, pure=False):
        self.buf = open(path, "rb").read()
        b = self.buf
        hsize, self.type, _, cert, _, tik, tmd, data, footer = struct.unpack_from(">IHHIIIIII", b, 0)
        o = _align(hsize)
        self.cert = b[o:o + cert]
        o += _align(cert)
        self.ticket = b[o:o + tik]
        o += _align(tik)
        self.tmd = b[o:o + tmd]
        o += _align(tmd)
        self.data_off = o
        self.footer = b[o + _align(data):o + _align(data) + footer]
        t = self.tmd
        self.ios = struct.unpack_from(">Q", t, 0x184)[0]
        self.title_id = t[0x18C:0x194]
        self.group = t[0x198:0x19A]
        self.region = struct.unpack_from(">H", t, 0x19C)[0]
        self.version = struct.unpack_from(">H", t, 0x1DC)[0]
        self.boot_index = struct.unpack_from(">H", t, 0x1E0)[0]
        self.contents = []
        for i in range(struct.unpack_from(">H", t, 0x1DE)[0]):
            cid, index, ctype, size = struct.unpack_from(">IHHQ", t, 0x1E4 + 36 * i)
            self.contents.append({"id": cid, "index": index, "type": ctype, "size": size,
                                  "sha1": t[0x1E4 + 36 * i + 16:0x1E4 + 36 * i + 36]})
        key = COMMON_KEYS.get(self.ticket[0x1F1], COMMON_KEYS[0])
        self.title_key = aes.cbc_decrypt(key, self.ticket[0x1DC:0x1E4] + bytes(8),
                                         self.ticket[0x1BF:0x1CF], pure)
        self.pure = pure

    @property
    def game_id(self):
        """The four-letter title code and the maker: what 0x80000000 holds."""
        return (self.title_id[4:] + self.group).decode("latin1")

    def read_content(self, c):
        """The decrypted content, checked against the TMD's SHA-1."""
        o = self.data_off
        for other in self.contents:
            if other is c:
                break
            o += _align(_align(other["size"], 16))
        enc = self.buf[o:o + _align(c["size"], 16)]
        dec = aes.cbc_decrypt(self.title_key, c["index"].to_bytes(2, "big") + bytes(14), enc, self.pure)
        dec = dec[:c["size"]]
        if hashlib.sha1(dec).digest() != c["sha1"]:
            raise ValueError(f"content {c['id']:08x}: SHA-1 mismatch")
        return dec

    def executable(self, contents=None):
        """(content, DOL bytes) of the game's executable, see the module's docstring."""
        contents = contents or {c["id"]: self.read_content(c) for c in self.contents}
        boot = next(c for c in self.contents if c["index"] == self.boot_index)
        found = []
        for c in self.contents:
            if c is boot:
                continue
            data = contents[c["id"]]
            dol = data if is_dol(data) else lz77_decompress(data)
            if dol and is_dol(dol):
                found.append((c, dol))
        if len(found) == 1:
            return found[0]
        if not found and is_dol(contents[boot["id"]]):
            return boot, contents[boot["id"]]
        raise ValueError(f"cannot tell the game's executable ({len(found)} DOLs beside the boot content)")


def extract(path, out, pure=False):
    w = Wad(path, pure)
    os.makedirs(os.path.join(out, "content"), exist_ok=True)
    os.makedirs(os.path.join(out, "sys"), exist_ok=True)
    for name, data in (("tmd.bin", w.tmd), ("ticket.bin", w.ticket), ("cert.bin", w.cert)):
        open(os.path.join(out, name), "wb").write(data)
    contents = {}
    for c in w.contents:
        contents[c["id"]] = w.read_content(c)
        open(os.path.join(out, "content", f"{c['id']:08x}.app"), "wb").write(contents[c["id"]])
    exe, dol = w.executable(contents)
    open(os.path.join(out, "sys", "main.dol"), "wb").write(dol)
    print(f"{w.game_id}: {len(w.contents)} contents -> {out}/content, "
          f"executable from content {exe['id']:08x} -> {out}/sys/main.dol ({len(dol)} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wad")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--extract", metavar="OUT")
    ap.add_argument("--pure", action="store_true", help="never use pycryptodome")
    a = ap.parse_args()
    if a.extract:
        extract(a.wad, a.extract, a.pure)
        return
    w = Wad(a.wad, a.pure)
    regions = {0: "Japan", 1: "USA", 2: "Europe", 3: "region free", 4: "Korea"}
    print(f"WAD '{w.type.to_bytes(2, 'big').decode('latin1')}'  {w.game_id}  title "
          f"{w.title_id[:4].hex()}-{w.title_id[4:].hex()}  v{w.version}  IOS{w.ios & 0xFFFFFFFF}  "
          f"{regions.get(w.region, w.region)}")
    for c in w.contents:
        boot = "  boot" if c["index"] == w.boot_index else ""
        print(f"  {c['index']:3d}  {c['id']:08x}  type {c['type']:04x}  {c['size']:10d}{boot}")


if __name__ == "__main__":
    main()
