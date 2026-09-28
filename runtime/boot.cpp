// wiikit runtime — what the system menu, IOS and the apploader leave behind
// before a disc game's __start: the executable loaded, the disc header, BI2
// and the FST at the top of MEM1, and the "low memory" globals at 0x80000000
// the SDK reads (memory sizes and arenas, clocks, the IOS version, the IPC
// buffer). Values follow Dolphin's HLE boot for an IOS of the 0x93600000
// memory layout (IOS 56 here, from the TMD). A GameCube disc finds what the
// GameCube's IPL leaves: the same first globals, its own clocks and the ARAM's
// size, and nothing of IOS.
#include "boot.h"
#include "disc.h"
#include "mem.h"
#include "rt.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> d;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return d;
    std::fseek(f, 0, SEEK_END);
    d.resize((size_t)std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    if (!d.empty() && std::fread(d.data(), 1, d.size(), f) != d.size()) d.clear();
    std::fclose(f);
    return d;
}

uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

// What IOS and the system leave in low memory for a Wii title (IOS 56 and
// later's memory map): the same for a disc and a NAND title, but the disc flag.
void ios_low_memory(uint32_t arena_hi, uint32_t ios, const uint8_t* id, bool disc) {
    const uint32_t ios_version = ios << 16 | 0x161D;
    st32(0x80003100, 0x01800000);                  // MEM1 physical size
    st32(0x80003104, 0x01800000);                  // MEM1 simulated size
    st32(0x80003108, 0x81800000);                  // MEM1 end
    st32(0x8000310C, 0);                           // MEM1 arena begin
    st32(0x80003110, arena_hi);                    // MEM1 arena end
    st32(0x80003114, 0xDEADBEEF);
    st32(0x80003118, 0x04000000);                  // MEM2 physical size
    st32(0x8000311C, 0x04000000);                  // MEM2 simulated size
    st32(0x80003120, 0x93600000);                  // MEM2 end
    st32(0x80003124, 0x90000800);                  // MEM2 arena begin
    st32(0x80003128, 0x935E0000);                  // MEM2 arena end
    st32(0x8000312C, 0xDEADBEEF);
    st32(0x80003130, 0x935E0000);                  // IPC buffer
    st32(0x80003134, 0x93600000);
    st32(0x80003138, 0x00000011);                  // Hollywood revision
    st32(0x8000313C, 0xDEADBEEF);
    st32(0x80003140, ios_version);
    st32(0x80003144, 0x03192009);                  // IOS build date
    st32(0x80003148, 0x93600000);                  // IOS reserved
    st32(0x8000314C, 0x93620000);
    st32(0x80003150, 0xDEADBEEF);
    st32(0x80003154, 0xDEADBEEF);
    st32(0x80003158, 0x0000FF16);                  // DDR vendor
    st8(0x8000315C, 0x80);                         // boot flag: booted by the system
    st8(0x8000315D, 0x00);
    st16(0x8000315E, 0x0113);                      // apploader version
    st32(0x80003160, 0);                           // system menu sync
    mem_write(0x80003180, id, 4);                  // game id, kept for the whole run
    st32(0x80003184, 0x80000000);
    st32(0x80003188, ios_version);                 // the IOS the game expects
    if (disc) st8(0x8000319C, 0x80);               // single-layer disc
}

}  // namespace

bool g_gamecube = false;
uint32_t g_bus_mhz = 243;

uint32_t boot_disc(const char* extract_dir, bool eurgb60) {
    std::string root = extract_dir;
    if (!disc_open(extract_dir)) rt_die("%s: no sys/boot.bin and sys/fst.bin", extract_dir);
    g_gamecube = disc_is_gamecube();
    if (g_gamecube) g_bus_mhz = 162;
    const int shift = g_gamecube ? 0 : 2;          // boot.bin's offsets: >> 2 on the Wii
    uint32_t entry = mem_load_dol((root + "/sys/main.dol").c_str());
    if (!entry) rt_die("%s/sys/main.dol: cannot load", extract_dir);
    const std::vector<uint8_t>& boot = disc_boot();
    std::vector<uint8_t> bi2 = slurp(root + "/sys/bi2.bin"), tmd = slurp(root + "/tmd.bin");
    const std::vector<uint8_t>& fst = disc_fst();

    // BI2 and the FST at the top of MEM1; the arena ends below them
    const uint32_t bi2_addr = 0x817FE000;
    const uint32_t fst_addr = (bi2_addr - (uint32_t)fst.size()) & ~0x1Fu;
    mem_write(bi2_addr, bi2.data(), std::min<size_t>(bi2.size(), 0x2000));
    mem_write(fst_addr, fst.data(), fst.size());

    mem_write(0x80000000, boot.data(), 0x20);      // game id, maker, disc, version, magic
    st32(0x80000020, 0x0D15EA5E);                  // booted by the system
    st32(0x80000024, 1);
    st32(0x80000028, 0x01800000);                  // MEM1 size
    st32(0x8000002C, g_gamecube ? 0x00000003 : 0x00000023);   // board: retail GameCube (HW2), retail Wii
    st32(0x80000030, 0);                           // arena low: the linker's
    st32(0x80000034, fst_addr);                    // arena high
    st32(0x80000038, fst_addr);                    // FST
    st32(0x8000003C, be32(&boot[0x42C]) << shift); // FST maximum size
    // video: the disc's region, as Dolphin boots it. A PAL game asks VI for
    // PAL, and the SDK refuses a switch from NTSC. With SYSCONF's IPL.E60 set
    // the IPL leaves the TV mode at EuRGB60 (5), which is what games read
    // (VIGetTvFormat) to choose 60 Hz over 50. The region is the disc's u32 at 0x4E000
    // (disc/region.bin: 0 Japan, 1 USA, 2 Europe, 4 Korea), else the game
    // id's fourth letter (E, J, K, W: NTSC; the others PAL)
    // (a GameCube disc: BI2's u32 at 0x18, same values)
    std::vector<uint8_t> reg = slurp(root + "/disc/region.bin");
    if (g_gamecube && bi2.size() >= 0x1C) reg.assign(bi2.begin() + 0x18, bi2.begin() + 0x1C);
    const bool pal = reg.size() >= 4 ? be32(reg.data()) == 2 :
                     !std::strchr("EJKW", boot[3]);
    st32(0x800000CC, pal ? (eurgb60 && !g_gamecube ? 5 : 1) : 0);   // the TV mode: 0 NTSC, 1 PAL, 5 EuRGB60
    hw_vi_preset(pal);
    st32(0x800000F0, 0x01800000);                  // simulated memory size
    st32(0x800000F4, bi2_addr);
    st32(0x800000F8, g_bus_mhz * 1000000);         // bus clock
    st32(0x800000FC, g_bus_mhz * 3000000);         // CPU clock: 729 MHz, 486 on the GameCube
    if (g_gamecube) {
        st32(0x800000D0, 0x01000000);              // ARAM size: 16 MB
        return entry;
    }

    ios_low_memory(fst_addr, tmd.size() >= 0x18C ? be32(&tmd[0x188]) : 56, boot.data(), true);
    return entry;
}

uint32_t boot_nand(const char* extract_dir, bool eurgb60) {
    std::string root = extract_dir;
    uint32_t entry = mem_load_dol((root + "/sys/main.dol").c_str());
    if (!entry) rt_die("%s/sys/main.dol: cannot load (python -m wiikit.wad TITLE.wad --extract)", extract_dir);
    std::vector<uint8_t> tmd = slurp(root + "/tmd.bin");
    if (tmd.size() < 0x1E4) rt_die("%s/tmd.bin: missing or short", extract_dir);

    // the header a disc would have: the title's four letters, the maker (the
    // TMD's group id), the Wii magic
    uint8_t header[0x20] = {};
    std::memcpy(header, &tmd[0x190], 4);
    std::memcpy(header + 4, &tmd[0x198], 2);
    header[0x18] = 0x5D; header[0x19] = 0x1C; header[0x1A] = 0x9E; header[0x1B] = 0xA3;

    // No disc, but the SDK's DVD code still looks for an FST: an empty one
    // (the root alone) where a disc's would be, the arena ending below it
    static const uint8_t fst[13] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    const uint32_t bi2_addr = 0x817FE000;
    const uint32_t fst_addr = (bi2_addr - (uint32_t)sizeof fst) & ~0x1Fu;
    mem_write(fst_addr, fst, sizeof fst);

    mem_write(0x80000000, header, sizeof header);
    st32(0x80000020, 0x0D15EA5E);                  // booted by the system
    st32(0x80000024, 1);
    st32(0x80000028, 0x01800000);                  // MEM1 size
    st32(0x8000002C, 0x00000023);                  // board: retail Wii
    st32(0x80000030, 0);                           // arena low: the linker's
    st32(0x80000034, fst_addr);                    // arena high
    st32(0x80000038, fst_addr);                    // FST
    st32(0x8000003C, sizeof fst);                  // FST maximum size
    // the region from the TMD (0 Japan, 1 USA, 2 Europe, 3 free, 4 Korea);
    // region free: the title code's fourth letter, as for a disc
    const uint16_t region = (uint16_t)(tmd[0x19C] << 8 | tmd[0x19D]);
    const bool pal = region == 2 || (region == 3 && !std::strchr("EJKW", header[3]));
    st32(0x800000CC, pal ? (eurgb60 ? 5 : 1) : 0); // the TV mode: 0 NTSC, 1 PAL, 5 EuRGB60
    hw_vi_preset(pal);
    st32(0x800000F0, 0x01800000);                  // simulated memory size
    st32(0x800000F4, bi2_addr);
    st32(0x800000F8, g_bus_mhz * 1000000);         // bus clock
    st32(0x800000FC, g_bus_mhz * 3000000);         // CPU clock

    ios_low_memory(fst_addr, be32(&tmd[0x188]), header, false);
    return entry;
}

std::string nand_title_name(const char* extract_dir) {
    // the banner is content index 0: IMET at 0x40 (0x80 in an installed
    // title's), then ten 42-character UTF-16 names (Japanese, English, ...)
    // from 0x1C past it
    std::string root = extract_dir;
    std::vector<uint8_t> tmd = slurp(root + "/tmd.bin");
    if (tmd.size() < 0x1E4 + 36) return {};
    const uint16_t count = (uint16_t)(tmd[0x1DE] << 8 | tmd[0x1DF]);
    for (uint16_t i = 0; i < count && 0x1E4 + 36u * (i + 1) <= tmd.size(); ++i) {
        const uint8_t* r = &tmd[0x1E4 + 36 * i];
        if (r[4] || r[5]) continue;                // index 0 only
        char name[32];
        std::snprintf(name, sizeof name, "/content/%08x.app", be32(r));
        std::vector<uint8_t> banner = slurp(root + name);
        for (size_t imet : {0x40u, 0x80u}) {
            const size_t english = imet + 0x1C + 84;
            if (banner.size() < english + 84 || std::memcmp(&banner[imet], "IMET", 4)) continue;
            std::string out;
            for (int k = 0; k < 42; ++k) {
                const uint16_t ch = (uint16_t)(banner[english + 2 * k] << 8 | banner[english + 2 * k + 1]);
                if (!ch) break;
                out += ch < 0x80 ? (char)ch : '?';
            }
            return out;
        }
        return {};
    }
    return {};
}
