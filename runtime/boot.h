// wiikit runtime — the state a disc game finds at __start (boot.cpp).
#pragma once
#include <cstdint>
#include <string>

// Load main.dol and set up memory as the system leaves it for a disc game,
// from a tree made by `python -m wiikit.disc GAME --extract DIR`. Returns
// the entry point.
// eurgb60: SYSCONF's IPL.E60, for a PAL disc: the IPL leaves VI in EuRGB60.
uint32_t boot_disc(const char* extract_dir, bool eurgb60 = false);

// Load sys/main.dol and set up memory as the system leaves it for a title
// installed on the NAND (WiiWare, Virtual Console, a channel), from a tree
// made by `python -m wiikit.wad TITLE.wad --extract DIR`: its contents are
// read through /dev/es (ios.cpp). Returns the entry point.
uint32_t boot_nand(const char* extract_dir, bool eurgb60 = false);
// The title's English name from its banner (content 0), or "".
std::string nand_title_name(const char* extract_dir);
