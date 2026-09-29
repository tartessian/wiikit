// wiikit runtime — a Gekko interpreter for code that is not in the executable.
//
// The recompiler translates what the executable holds. Some games also run
// code they write at run time: a JIT (Mono's, under Unity), code copied or
// unpacked into memory, trampolines. ppc_call_indirect() sends a call to an
// address with no recompiled function here, and the interpreter runs it
// against the same PPCContext, with the same semantics: every instruction
// does what recompile/emit.py makes of it, through the same helpers of ppc.h.
//
// Control flow follows the recompiled code's conventions, so that the two
// kinds of code call each other freely:
//   * a call (bl, bcl, bctrl, blrl) goes through ppc_call_indirect: to the
//     recompiled function when there is one, else to the interpreter;
//     `bcl 20,31,$+4` (and any call to the next instruction) only sets LR
//   * blr returns, as a recompiled function's does
//   * a jump without link (b, bc, bctr) to a recompiled function is a tail
//     call: call it, then return; to any other address it continues here
//   * backward branches poll for interrupts, as recompiled loops do
//
// Straight runs of code (up to a branch) are decoded once into blocks: the
// frequent instructions with their fields worked out and a handler each,
// dispatched one to the next (computed goto, so that each handler's jump has
// its own history in the branch predictor), the rest through step(), which
// knows every instruction. A block keeps the words it was decoded from and
// is checked against memory each time it is entered, so code that is
// rewritten needs nothing from its writer here, as before.
#include "rt.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cstddef>
#include <vector>

namespace {

inline uint32_t mask32(uint32_t mb, uint32_t me) {
    uint32_t m = 0;
    for (uint32_t i = mb;; i = (i + 1) & 31) {
        m |= 1u << (31 - i);
        if (i == me) break;
    }
    return m;
}

inline uint32_t s16(uint32_t w) { return (uint32_t)(int32_t)(int16_t)(w & 0xFFFF); }
inline uint32_t s12(uint32_t w) { return (uint32_t)((int32_t)(w << 20) >> 20); }

// BO/BI: decrements CTR when BO says so, and says whether the branch is taken
inline bool bc_taken(PPCContext& c, uint32_t bo, uint32_t bi) {
    bool ok = true;
    if (!(bo & 4)) {
        c.ctr--;
        ok = (bo & 2) ? c.ctr == 0 : c.ctr != 0;
    }
    if (!(bo & 16)) ok = ok && ((bo & 8) ? c.cr[bi] != 0 : c.cr[bi] == 0);
    return ok;
}

// SPRs of the processor, not of a thread (recompile/emit.py's _GLOBAL_SPRS)
inline bool global_spr(uint32_t n) {
    return n == 920 || n == 921 || n == 922 || n == 923 || n == 1008 || n == 1009 || n == 1011 || n == 1017;
}

[[noreturn]] void bad(PPCContext& c, uint32_t pc, uint32_t w) {
    rt_die("interp: %s: unsupported instruction %08X", rt_name(pc).c_str(), w);
}

}  // namespace

namespace {

enum Flow : uint8_t { F_NEXT, F_JUMP, F_RETURN };
struct Step { Flow flow; uint32_t target; };

// One instruction at pc, word w: on to the next, a jump without link (to
// `target`), or a return
Step step(PPCContext& c, const uint32_t pc, const uint32_t w) {
    {
        const uint32_t nxt = pc + 4;
        const uint32_t op = w >> 26;
        const uint32_t D = (w >> 21) & 31, A = (w >> 16) & 31, B = (w >> 11) & 31, C = (w >> 6) & 31;
        const bool rc = w & 1;
        uint32_t target = 0;
        bool jump = false;                            // to `target`, without link

        switch (op) {
        // ---- branches ----------------------------------------------------------------
        case 18: {                                    // b
            uint32_t li = w & 0x03FFFFFC;
            if (li & 0x02000000) li |= 0xFC000000;
            const uint32_t t = (w & 2) ? li : pc + li;
            if (w & 1) {
                c.lr = nxt;
                if (t != nxt) ppc_call_indirect(c, t);
            } else {
                target = t;
                jump = true;
            }
            break;
        }
        case 16: {                                    // bc
            uint32_t bd = w & 0xFFFC;
            if (bd & 0x8000) bd |= 0xFFFF0000;
            const uint32_t t = (w & 2) ? bd : pc + bd;
            if (!bc_taken(c, D, A)) break;
            if (w & 1) {
                c.lr = nxt;
                if (t != nxt) ppc_call_indirect(c, t);
            } else {
                target = t;
                jump = true;
            }
            break;
        }
        case 19: {
            const uint32_t xo = (w >> 1) & 0x3FF;
            switch (xo) {
            case 16:                                  // bclr
                if (!bc_taken(c, D, A)) break;
                if (w & 1) {
                    const uint32_t t = c.lr;
                    c.lr = nxt;
                    ppc_call_indirect(c, t);
                    break;
                }
                return {F_RETURN, 0};
            case 528:                                 // bcctr
                if (!bc_taken(c, D, A)) break;
                if (w & 1) {
                    c.lr = nxt;
                    ppc_call_indirect(c, c.ctr);
                } else {
                    target = c.ctr;
                    jump = true;
                }
                break;
            case 0:                                   // mcrf
                for (int k = 0; k < 4; ++k) c.cr[((w >> 23) & 7) * 4 + k] = c.cr[((w >> 18) & 7) * 4 + k];
                break;
            case 150: break;                          // isync
            case 257: c.cr[D] = (uint8_t)((c.cr[A] & c.cr[B]) & 1); break;      // crand
            case 129: c.cr[D] = (uint8_t)((c.cr[A] & !c.cr[B]) & 1); break;     // crandc
            case 449: c.cr[D] = (uint8_t)((c.cr[A] | c.cr[B]) & 1); break;      // cror
            case 417: c.cr[D] = (uint8_t)((c.cr[A] | !c.cr[B]) & 1); break;     // crorc
            case 193: c.cr[D] = (uint8_t)((c.cr[A] ^ c.cr[B]) & 1); break;      // crxor
            case 33: c.cr[D] = (uint8_t)((!(c.cr[A] | c.cr[B])) & 1); break;      // crnor
            case 225: c.cr[D] = (uint8_t)((!(c.cr[A] & c.cr[B])) & 1); break;     // crnand
            case 289: c.cr[D] = (uint8_t)((!(c.cr[A] ^ c.cr[B])) & 1); break;     // creqv
            case 50:                                  // rfi
                ppc_unimplemented(c, pc, "rfi");
                return {F_RETURN, 0};
            default: bad(c, pc, w);
            }
            break;
        }
        case 17:                                      // sc
            if (!(w & 2)) bad(c, pc, w);
            ppc_syscall(c, pc);
            break;

        // ---- integer, immediate ------------------------------------------------------
        case 14: c.r[D] = (A ? c.r[A] : 0) + s16(w); break;                              // addi
        case 15: c.r[D] = (A ? c.r[A] : 0) + (w << 16); break;                           // addis
        case 12: c.r[D] = add_ca(c, c.r[A], s16(w), 0); break;                           // addic
        case 13: c.r[D] = add_ca(c, c.r[A], s16(w), 0); cr0_rc(c, c.r[D]); break;        // addic.
        case 8: c.r[D] = add_ca(c, ~c.r[A], s16(w), 1); break;                           // subfic
        case 7: c.r[D] = c.r[A] * s16(w); break;                                          // mulli
        case 24: c.r[A] = c.r[D] | (w & 0xFFFF); break;                                   // ori
        case 25: c.r[A] = c.r[D] | (w << 16); break;                                      // oris
        case 26: c.r[A] = c.r[D] ^ (w & 0xFFFF); break;                                   // xori
        case 27: c.r[A] = c.r[D] ^ (w << 16); break;                                      // xoris
        case 28: c.r[A] = c.r[D] & (w & 0xFFFF); cr0_rc(c, c.r[A]); break;               // andi.
        case 29: c.r[A] = c.r[D] & (w << 16); cr0_rc(c, c.r[A]); break;                  // andis.
        case 11: cr_cmp_s(c, (w >> 23) & 7, (int32_t)c.r[A], (int32_t)s16(w)); break;    // cmpi
        case 10: cr_cmp_u(c, (w >> 23) & 7, c.r[A], w & 0xFFFF); break;                  // cmpli
        case 3:                                                                           // twi
            if (D == 31 || trap_cond(D, c.r[A], s16(w))) ppc_trap(c, pc);
            break;

        // ---- rotates -----------------------------------------------------------------
        case 20: {                                                                        // rlwimi
            const uint32_t m = mask32(C, (w >> 1) & 31);
            c.r[A] = (rotl32(c.r[D], B) & m) | (c.r[A] & ~m);
            if (rc) cr0_rc(c, c.r[A]);
            break;
        }
        case 21:                                                                          // rlwinm
            c.r[A] = rotl32(c.r[D], B) & mask32(C, (w >> 1) & 31);
            if (rc) cr0_rc(c, c.r[A]);
            break;
        case 23:                                                                          // rlwnm
            c.r[A] = rotl32(c.r[D], c.r[B]) & mask32(C, (w >> 1) & 31);
            if (rc) cr0_rc(c, c.r[A]);
            break;

        // ---- loads and stores, D-form ------------------------------------------------
        case 32: case 33: case 34: case 35: case 40: case 41: case 42: case 43: {         // lwz lbz lhz lha (u)
            const bool upd = op & 1;
            const uint32_t ea = (upd || A ? c.r[A] : 0) + s16(w);
            uint32_t v;
            switch (op & ~1u) {
            case 32: v = ld32(ea); break;
            case 34: v = ld8(ea); break;
            case 40: v = ld16(ea); break;
            default: v = (uint32_t)(int32_t)(int16_t)ld16(ea); break;
            }
            c.r[D] = v;
            if (upd) c.r[A] = ea;
            break;
        }
        case 36: case 37: case 38: case 39: case 44: case 45: {                          // stw stb sth (u)
            const bool upd = op & 1;
            const uint32_t ea = (upd || A ? c.r[A] : 0) + s16(w);
            switch (op & ~1u) {
            case 36: st32(ea, c.r[D]); break;
            case 38: st8(ea, (uint8_t)c.r[D]); break;
            default: st16(ea, (uint16_t)c.r[D]); break;
            }
            if (upd) c.r[A] = ea;
            break;
        }
        case 46: {                                                                        // lmw
            const uint32_t ea = (A ? c.r[A] : 0) + s16(w);
            for (uint32_t r = D, k = 0; r < 32; ++r, ++k) c.r[r] = ld32(ea + 4 * k);
            break;
        }
        case 47: {                                                                        // stmw
            const uint32_t ea = (A ? c.r[A] : 0) + s16(w);
            for (uint32_t r = D, k = 0; r < 32; ++r, ++k) st32(ea + 4 * k, c.r[r]);
            break;
        }
        case 48: case 49: case 50: case 51: case 52: case 53: case 54: case 55: {         // lfs lfd stfs stfd (u)
            const bool upd = op & 1;
            const uint32_t ea = (upd || A ? c.r[A] : 0) + s16(w);
            switch (op & ~1u) {
            case 48: { double v = (double)as_f32(ld32(ea)); c.f[D] = v; c.ps1[D] = v; break; }
            case 50: c.f[D] = as_f64(ld64(ea)); break;
            case 52: st32(ea, as_u32((float)c.f[D])); break;
            default: st64(ea, as_u64(c.f[D])); break;
            }
            if (upd) c.r[A] = ea;
            break;
        }
        case 56: case 57: case 60: case 61: {                                             // psq_l psq_st (u)
            const bool upd = op & 1;
            const uint32_t ea = (upd || A ? c.r[A] : 0) + s12(w);
            const int W = (w >> 15) & 1, I = (w >> 12) & 7;
            if (op < 60) psq_load(c, D, ea, W, I); else psq_store(c, D, ea, W, I);
            if (upd) c.r[A] = ea;
            break;
        }

        // ---- opcode 31 -----------------------------------------------------------------
        case 31: {
            const uint32_t xo = (w >> 1) & 0x3FF, x9 = xo & 0x1FF;
            const uint32_t eax = (A ? c.r[A] : 0) + c.r[B];
            // XO-form arithmetic (the OE bit is xo's top bit); mulhw(u) have no OE form
            const bool arith = x9 == 8 || x9 == 10 || x9 == 11 || x9 == 40 || x9 == 75 || x9 == 104 ||
                               x9 == 136 || x9 == 138 || x9 == 200 || x9 == 202 || x9 == 232 ||
                               x9 == 234 || x9 == 235 || x9 == 266 || x9 == 459 || x9 == 491;
            if (arith && xo != 11 + 512 && xo != 75 + 512) {
                const uint32_t a = c.r[A], b = c.r[B];
                uint32_t r = 0;
                bool ov = false;
                switch (x9) {
                case 266: r = a + b; ov = ((a ^ r) & (b ^ r)) >> 31; break;                        // add
                case 40: r = b - a; ov = ((~a ^ r) & (b ^ r)) >> 31; break;                        // subf
                case 104: r = 0u - a; ov = a == 0x80000000u; break;                                 // neg
                case 235: {                                                                          // mullw
                    int64_t p = (int64_t)(int32_t)a * (int32_t)b;
                    r = (uint32_t)p;
                    ov = p != (int64_t)(int32_t)r;
                    break;
                }
                case 75: r = (uint32_t)(((int64_t)(int32_t)a * (int32_t)b) >> 32); break;          // mulhw
                case 11: r = (uint32_t)(((uint64_t)a * b) >> 32); break;                            // mulhwu
                case 491: r = divw(a, b); ov = b == 0 || (a == 0x80000000u && b == 0xFFFFFFFFu); break;
                case 459: r = divwu(a, b); ov = b == 0; break;
                default: {                                                                           // carrying adds
                    uint32_t x, y, cin;
                    switch (x9) {
                    case 10: x = a; y = b; cin = 0; break;                             // addc
                    case 138: x = a; y = b; cin = c.xer_ca; break;                     // adde
                    case 202: x = a; y = 0; cin = c.xer_ca; break;                     // addze
                    case 234: x = a; y = 0xFFFFFFFFu; cin = c.xer_ca; break;           // addme
                    case 8: x = ~a; y = b; cin = 1; break;                             // subfc
                    case 136: x = ~a; y = b; cin = c.xer_ca; break;                    // subfe
                    case 200: x = ~a; y = 0; cin = c.xer_ca; break;                    // subfze
                    default: x = ~a; y = 0xFFFFFFFFu; cin = c.xer_ca; break;           // subfme
                    }
                    r = add_ca(c, x, y, cin);
                    ov = ((x ^ r) & (y ^ r)) >> 31;
                    break;
                }
                }
                c.r[D] = r;
                if (w & 0x400) set_ov(c, ov);
                if (rc) cr0_rc(c, r);
                break;
            }
            switch (xo) {
            case 0: cr_cmp_s(c, (w >> 23) & 7, (int32_t)c.r[A], (int32_t)c.r[B]); break;    // cmp
            case 32: cr_cmp_u(c, (w >> 23) & 7, c.r[A], c.r[B]); break;                     // cmpl
            case 4: if (D == 31 || trap_cond(D, c.r[A], c.r[B])) ppc_trap(c, pc); break;    // tw
            case 19: c.r[D] = mfcr(c); break;                                                // mfcr
            case 83: c.r[D] = c.msr; break;                                                  // mfmsr
            case 146: ppc_mtmsr(c, c.r[D]); break;                                           // mtmsr
            case 144: mtcrf(c, (w >> 12) & 0xFF, c.r[D]); break;                            // mtcrf
            case 339: {                                                                      // mfspr
                const uint32_t n = A | B << 5;
                uint32_t v;
                if (global_spr(n)) v = g_ppc_spr[n];
                else if (n == 1) v = mfxer(c);
                else if (n == 8) v = c.lr;
                else if (n == 9) v = c.ctr;
                else if (n == 22) v = ppc_mfdec();
                else if (n == 268) v = (uint32_t)ppc_timebase();
                else if (n == 269) v = (uint32_t)(ppc_timebase() >> 32);
                else if (n >= 912 && n < 920) v = c.gqr[n - 912];
                else v = c.spr[n];
                c.r[D] = v;
                break;
            }
            case 467: {                                                                      // mtspr
                const uint32_t n = A | B << 5, v = c.r[D];
                if (n == 1) mtxer(c, v);
                else if (n == 22) ppc_mtdec(v);
                else if (n == 284 || n == 285) ppc_mttb(n - 284, v);
                else if (n == 921) { g_ppc_spr[921] = v; ppc_wpar_write(v); }
                else if (n == 923) g_ppc_spr[923] = ppc_lc_dma(g_ppc_spr[922], v);
                else if (global_spr(n)) g_ppc_spr[n] = v;
                else if (n == 8) c.lr = v;
                else if (n == 9) c.ctr = v;
                else if (n >= 912 && n < 920) c.gqr[n - 912] = v;
                else c.spr[n] = v;
                break;
            }
            case 371: c.r[D] = (uint32_t)(ppc_timebase() >> ((A | B << 5) == 269 ? 32 : 0)); break;   // mftb
            case 210: c.sr[A & 15] = c.r[D]; break;                                          // mtsr
            case 595: c.r[D] = c.sr[A & 15]; break;                                          // mfsr
            case 242: c.sr[c.r[B] >> 28] = c.r[D]; break;                                    // mtsrin
            case 659: c.r[D] = c.sr[c.r[B] >> 28]; break;                                    // mfsrin
            case 512: {                                                                      // mcrxr
                cr_set(c, (w >> 23) & 7, c.xer_so, c.xer_ov, c.xer_ca, 0);
                c.xer_so = c.xer_ov = c.xer_ca = 0;
                break;
            }
            // logic and shifts: rA = f(rS, rB)
            case 28: c.r[A] = c.r[D] & c.r[B]; if (rc) cr0_rc(c, c.r[A]); break;            // and
            case 60: c.r[A] = c.r[D] & ~c.r[B]; if (rc) cr0_rc(c, c.r[A]); break;           // andc
            case 444: c.r[A] = c.r[D] | c.r[B]; if (rc) cr0_rc(c, c.r[A]); break;           // or
            case 412: c.r[A] = c.r[D] | ~c.r[B]; if (rc) cr0_rc(c, c.r[A]); break;          // orc
            case 316: c.r[A] = c.r[D] ^ c.r[B]; if (rc) cr0_rc(c, c.r[A]); break;           // xor
            case 124: c.r[A] = ~(c.r[D] | c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;        // nor
            case 476: c.r[A] = ~(c.r[D] & c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;        // nand
            case 284: c.r[A] = ~(c.r[D] ^ c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;        // eqv
            case 24: c.r[A] = slw(c.r[D], c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;        // slw
            case 536: c.r[A] = srw(c.r[D], c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;       // srw
            case 792: c.r[A] = sraw(c, c.r[D], c.r[B]); if (rc) cr0_rc(c, c.r[A]); break;   // sraw
            case 824: c.r[A] = sraw(c, c.r[D], B); if (rc) cr0_rc(c, c.r[A]); break;        // srawi
            case 26: c.r[A] = cntlzw(c.r[D]); if (rc) cr0_rc(c, c.r[A]); break;             // cntlzw
            case 922: c.r[A] = (uint32_t)(int32_t)(int16_t)c.r[D]; if (rc) cr0_rc(c, c.r[A]); break;   // extsh
            case 954: c.r[A] = (uint32_t)(int32_t)(int8_t)c.r[D]; if (rc) cr0_rc(c, c.r[A]); break;    // extsb
            // indexed loads and stores
            case 23: c.r[D] = ld32(eax); break;                                              // lwzx
            case 55: c.r[D] = ld32(eax); c.r[A] = eax; break;                                // lwzux
            case 87: c.r[D] = ld8(eax); break;                                               // lbzx
            case 119: c.r[D] = ld8(eax); c.r[A] = eax; break;                                // lbzux
            case 279: c.r[D] = ld16(eax); break;                                             // lhzx
            case 311: c.r[D] = ld16(eax); c.r[A] = eax; break;                               // lhzux
            case 343: c.r[D] = (uint32_t)(int32_t)(int16_t)ld16(eax); break;                 // lhax
            case 375: c.r[D] = (uint32_t)(int32_t)(int16_t)ld16(eax); c.r[A] = eax; break;   // lhaux
            case 534: c.r[D] = PPC_BSWAP32(ld32(eax)); break;                                // lwbrx
            case 790: c.r[D] = (uint32_t)PPC_BSWAP16(ld16(eax)); break;                      // lhbrx
            case 20: c.reserve = eax; c.r[D] = ld32(eax); break;                             // lwarx
            case 151: st32(eax, c.r[D]); break;                                              // stwx
            case 183: st32(eax, c.r[D]); c.r[A] = eax; break;                                // stwux
            case 215: st8(eax, (uint8_t)c.r[D]); break;                                      // stbx
            case 247: st8(eax, (uint8_t)c.r[D]); c.r[A] = eax; break;                        // stbux
            case 407: st16(eax, (uint16_t)c.r[D]); break;                                    // sthx
            case 439: st16(eax, (uint16_t)c.r[D]); c.r[A] = eax; break;                      // sthux
            case 662: st32(eax, PPC_BSWAP32(c.r[D])); break;                                 // stwbrx
            case 918: st16(eax, PPC_BSWAP16((uint16_t)c.r[D])); break;                       // sthbrx
            case 150: st32(eax, c.r[D]); cr_set(c, 0, 0, 0, 1, c.xer_so); break;             // stwcx.
            case 597: ppc_lswx(c, D, A ? c.r[A] : 0, B ? B : 32); break;                     // lswi
            case 725: ppc_stswx(c, D, A ? c.r[A] : 0, B ? B : 32); break;                    // stswi
            case 533: ppc_lswx(c, D, eax, c.xer_bc); break;                                  // lswx
            case 661: ppc_stswx(c, D, eax, c.xer_bc); break;                                 // stswx
            case 535: { double v = (double)as_f32(ld32(eax)); c.f[D] = v; c.ps1[D] = v; break; }   // lfsx
            case 567: { double v = (double)as_f32(ld32(eax)); c.f[D] = v; c.ps1[D] = v; c.r[A] = eax; break; }
            case 599: c.f[D] = as_f64(ld64(eax)); break;                                     // lfdx
            case 631: c.f[D] = as_f64(ld64(eax)); c.r[A] = eax; break;                       // lfdux
            case 663: st32(eax, as_u32((float)c.f[D])); break;                               // stfsx
            case 695: st32(eax, as_u32((float)c.f[D])); c.r[A] = eax; break;                 // stfsux
            case 727: st64(eax, as_u64(c.f[D])); break;                                      // stfdx
            case 759: st64(eax, as_u64(c.f[D])); c.r[A] = eax; break;                        // stfdux
            case 983: st32(eax, (uint32_t)as_u64(c.f[D])); break;                            // stfiwx
            case 1014: ppc_dcbz(eax); break;                                                 // dcbz
            // caches, ordering, the TLB: nothing to do
            case 54: case 86: case 246: case 278: case 470: case 982: case 306: case 566: case 598: case 854:
                break;
            case 310: case 438:                                                              // eciwx, ecowx
                ppc_unimplemented(c, pc, xo == 310 ? "eciwx" : "ecowx");
                break;
            default: bad(c, pc, w);
            }
            break;
        }

        // ---- floating point --------------------------------------------------------------
        case 59: {
            const double a = c.f[A], b = c.f[B], cc = c.f[C];
            switch ((w >> 1) & 31) {
            case 21: fp_single(c, D, a + b); break;                    // fadds
            case 20: fp_single(c, D, a - b); break;                    // fsubs
            case 18: fp_single(c, D, a / b); break;                    // fdivs
            case 25: fp_single(c, D, a * cc); break;                   // fmuls
            case 29: fp_single(c, D, a * cc + b); break;               // fmadds
            case 28: fp_single(c, D, a * cc - b); break;               // fmsubs
            case 31: fp_single(c, D, -(a * cc + b)); break;            // fnmadds
            case 30: fp_single(c, D, -(a * cc - b)); break;            // fnmsubs
            case 22: fp_single(c, D, std::sqrt(b)); break;             // fsqrts
            case 24: fp_single(c, D, fres(b)); break;                  // fres
            default: bad(c, pc, w);
            }
            if (rc) cr1_rc(c);
            break;
        }
        case 63: {
            const uint32_t xo = (w >> 1) & 0x3FF;
            switch (xo) {
            case 0: case 32: fcmp(c, (w >> 23) & 7, c.f[A], c.f[B]); break;                  // fcmpu, fcmpo
            case 12: fp_single(c, D, c.f[B]); break;                                         // frsp
            case 14: c.f[D] = fctiw_bits(c, c.f[B], false); break;                           // fctiw
            case 15: c.f[D] = fctiw_bits(c, c.f[B], true); break;                            // fctiwz
            case 40: c.f[D] = -c.f[B]; break;                                                // fneg
            case 72: c.f[D] = c.f[B]; break;                                                 // fmr
            case 136: c.f[D] = -std::fabs(c.f[B]); break;                                    // fnabs
            case 264: c.f[D] = std::fabs(c.f[B]); break;                                     // fabs
            case 38: c.fpscr |= 0x80000000u >> D; break;                                     // mtfsb1
            case 70: c.fpscr &= ~(0x80000000u >> D); break;                                  // mtfsb0
            case 64: {                                                                       // mcrfs
                const uint32_t v = c.fpscr >> (28 - 4 * ((w >> 18) & 7));
                cr_set(c, (w >> 23) & 7, v >> 3 & 1, v >> 2 & 1, v >> 1 & 1, v & 1);
                break;
            }
            case 134: {                                                                      // mtfsfi
                const uint32_t sh = 28 - 4 * ((w >> 23) & 7);
                c.fpscr = (c.fpscr & ~(0xFu << sh)) | (((w >> 12) & 15) << sh);
                break;
            }
            case 583: c.f[D] = as_f64(0xFFF8000000000000ull | c.fpscr); break;               // mffs
            case 711: mtfsf(c, (w >> 17) & 0xFF, c.f[B]); break;                             // mtfsf
            default: {
                const double a = c.f[A], b = c.f[B], cc = c.f[C];
                switch (xo & 31) {
                case 21: c.f[D] = a + b; break;                        // fadd
                case 20: c.f[D] = a - b; break;                        // fsub
                case 18: c.f[D] = a / b; break;                        // fdiv
                case 25: c.f[D] = a * cc; break;                       // fmul
                case 29: c.f[D] = a * cc + b; break;                   // fmadd
                case 28: c.f[D] = a * cc - b; break;                   // fmsub
                case 31: c.f[D] = -(a * cc + b); break;                // fnmadd
                case 30: c.f[D] = -(a * cc - b); break;                // fnmsub
                case 23: c.f[D] = a >= 0.0 ? cc : b; break;            // fsel
                case 22: c.f[D] = std::sqrt(b); break;                 // fsqrt
                case 26: c.f[D] = frsqrte(b); break;                   // frsqrte
                default: bad(c, pc, w);
                }
            }
            }
            if (rc && xo != 0 && xo != 32 && xo != 64) cr1_rc(c);
            break;
        }

        // ---- paired singles ------------------------------------------------------------
        case 4: {
            const uint32_t xo = (w >> 1) & 0x3FF;
            const double a0 = c.f[A], a1 = c.ps1[A], b0 = c.f[B], b1 = c.ps1[B], c0 = c.f[C], c1 = c.ps1[C];
            switch (xo) {
            case 0: case 32: fcmp(c, (w >> 23) & 7, a0, b0); goto ps_done;                  // ps_cmpu0/o0
            case 64: case 96: fcmp(c, (w >> 23) & 7, a1, b1); goto ps_done;                 // ps_cmpu1/o1
            case 40: c.f[D] = -b0; c.ps1[D] = -b1; goto ps_rc;                               // ps_neg
            case 72: c.f[D] = b0; c.ps1[D] = b1; goto ps_rc;                                 // ps_mr
            case 136: c.f[D] = -std::fabs(b0); c.ps1[D] = -std::fabs(b1); goto ps_rc;       // ps_nabs
            case 264: c.f[D] = std::fabs(b0); c.ps1[D] = std::fabs(b1); goto ps_rc;         // ps_abs
            case 528: c.f[D] = a0; c.ps1[D] = b0; goto ps_rc;                                // ps_merge00
            case 560: c.f[D] = a0; c.ps1[D] = b1; goto ps_rc;                                // ps_merge01
            case 592: c.f[D] = a1; c.ps1[D] = b0; goto ps_rc;                                // ps_merge10
            case 624: c.f[D] = a1; c.ps1[D] = b1; goto ps_rc;                                // ps_merge11
            case 1014: ppc_dcbz((A ? c.r[A] : 0) + c.r[B]); goto ps_done;                   // dcbz_l
            }
            switch (xo & 0x3F) {                                                             // psq_lx psq_stx (u)
            case 6: case 7: case 38: case 39: {
                const uint32_t ea = (A ? c.r[A] : 0) + c.r[B];
                const int W = (w >> 10) & 1, I = (w >> 7) & 7;
                if ((xo & 1) == 0) psq_load(c, D, ea, W, I); else psq_store(c, D, ea, W, I);
                if (xo & 32) c.r[A] = ea;
                goto ps_done;
            }
            }
            switch (xo & 31) {
            case 21: ps_set(c, D, a0 + b0, a1 + b1); break;                        // ps_add
            case 20: ps_set(c, D, a0 - b0, a1 - b1); break;                        // ps_sub
            case 25: ps_set(c, D, a0 * c0, a1 * c1); break;                        // ps_mul
            case 18: ps_set(c, D, a0 / b0, a1 / b1); break;                        // ps_div
            case 29: ps_set(c, D, a0 * c0 + b0, a1 * c1 + b1); break;              // ps_madd
            case 28: ps_set(c, D, a0 * c0 - b0, a1 * c1 - b1); break;              // ps_msub
            case 31: ps_set(c, D, -(a0 * c0 + b0), -(a1 * c1 + b1)); break;        // ps_nmadd
            case 30: ps_set(c, D, -(a0 * c0 - b0), -(a1 * c1 - b1)); break;        // ps_nmsub
            case 23: ps_set(c, D, a0 >= 0.0 ? c0 : b0, a1 >= 0.0 ? c1 : b1); break;   // ps_sel
            case 24: ps_set(c, D, 1.0 / b0, 1.0 / b1); break;                      // ps_res
            case 26: ps_set(c, D, 1.0 / std::sqrt(b0), 1.0 / std::sqrt(b1)); break;   // ps_rsqrte
            case 10: ps_set(c, D, a0 + b1, c1); break;                             // ps_sum0
            case 11: ps_set(c, D, c0, a0 + b1); break;                             // ps_sum1
            case 12: ps_set(c, D, a0 * c0, a1 * c0); break;                        // ps_muls0
            case 13: ps_set(c, D, a0 * c1, a1 * c1); break;                        // ps_muls1
            case 14: ps_set(c, D, a0 * c0 + b0, a1 * c0 + b1); break;              // ps_madds0
            case 15: ps_set(c, D, a0 * c1 + b0, a1 * c1 + b1); break;              // ps_madds1
            default: bad(c, pc, w);
            }
        ps_rc:
            if (rc) cr1_rc(c);
        ps_done:
            break;
        }

        default:
            bad(c, pc, w);
        }

        return jump ? Step{F_JUMP, target} : Step{F_NEXT, 0};
    }
}

// ---- blocks ----------------------------------------------------------------------------------
enum Kind : uint16_t {
    K_END,                                         // past the block's last instruction
    K_STEP,                                        // step(), then on in the block
    K_TERM,                                        // step(), and the block ends: branches, and what may run guest code
    K_LI, K_ADDI, K_ORI, K_XORI, K_ANDI, K_CMPWI, K_CMPLWI, K_RLWINM,
    K_LWZ, K_LBZ, K_LHZ, K_LHA, K_STW, K_STB, K_STH, K_STWU,
    K_LFS, K_LFD, K_STFS, K_STFD,
    K_OR, K_AND, K_XOR, K_ADD, K_SUBF, K_NEG, K_MULLW, K_SLW, K_SRW, K_SRAWI, K_EXTSH, K_EXTSB,
    K_CMPW, K_CMPLW, K_LWZX, K_STWX, K_LBZX, K_STBX,
    K_MFLR, K_MTLR, K_MFCTR, K_MTCTR,
    K_FMR, K_FNEG, K_FADD, K_FSUB, K_FMUL, K_FADDS, K_FSUBS, K_FMULS, K_FMADDS, K_FMSUBS, K_FCMPU, K_FRSP,
    K_ADDIC,
    K_B, K_BL, K_BC, K_BLR,                        // branches (the block ends): imm is the target
    K_COUNT
};

struct Op {
    uint16_t kind;
    uint8_t d, a, b, cc;                           // cc: C, or a CR field
    uint32_t imm;                                  // an immediate, a mask
    uint32_t w;
};

struct Block {
    uint32_t start, n;                             // n instructions, their words after the ops
    // the block's branch to a fixed address (b, bl, bc): what ppc_lookup said
    // of it, for as long as its generation holds; to_interp: code written at
    // run time with no native, straight to interp_call
    uint32_t gen;
    bool to_interp;
    PPCFunc fn;
    Op ops[1];                                     // n + 1: a K_END closes them
    const uint8_t* words() const { return reinterpret_cast<const uint8_t*>(ops + n + 1); }
};

inline bool ends_block(uint16_t kind) { return kind == K_TERM || (kind >= K_B && kind <= K_BLR); }

// The kind of the instruction with word w at pc, and its fields
Op classify(uint32_t w, uint32_t pc) {
    const uint32_t op = w >> 26, D = (w >> 21) & 31, A = (w >> 16) & 31, B = (w >> 11) & 31, C = (w >> 6) & 31;
    const bool rc = w & 1;
    Op o{K_STEP, (uint8_t)D, (uint8_t)A, (uint8_t)B, (uint8_t)C, 0, w};
    auto k = [&](Kind kind, uint32_t imm = 0) { o.kind = kind; o.imm = imm; return o; };
    switch (op) {
    case 18: {                                                              // b, bl
        uint32_t li = w & 0x03FFFFFC;
        if (li & 0x02000000) li |= 0xFC000000;
        const uint32_t t = (w & 2) ? li : pc + li;
        return k((w & 1) ? K_BL : K_B, t);
    }
    case 16: {                                                              // bc (bcl: step())
        if (w & 1) return k(K_TERM);
        uint32_t bd = w & 0xFFFC;
        if (bd & 0x8000) bd |= 0xFFFF0000;
        return k(K_BC, (w & 2) ? bd : pc + bd);
    }
    case 19:                                                                // blr; the rest of 19: step()
        return ((w >> 1) & 0x3FF) == 16 && !(w & 1) && (D & 0x14) == 0x14 ? k(K_BLR) : k(K_TERM);
    case 17: case 3: return k(K_TERM);                                      // sc, twi
    case 12: return k(K_ADDIC, s16(w));
    case 14: return A ? k(K_ADDI, s16(w)) : k(K_LI, s16(w));
    case 15: return A ? k(K_ADDI, w << 16) : k(K_LI, w << 16);
    case 24: return k(K_ORI, w & 0xFFFF);
    case 25: return k(K_ORI, w << 16);
    case 26: return k(K_XORI, w & 0xFFFF);
    case 27: return k(K_XORI, w << 16);
    case 28: return k(K_ANDI, w & 0xFFFF);
    case 29: return k(K_ANDI, w << 16);
    case 11: o.cc = (uint8_t)((w >> 23) & 7); return k(K_CMPWI, s16(w));
    case 10: o.cc = (uint8_t)((w >> 23) & 7); return k(K_CMPLWI, w & 0xFFFF);
    case 21: return rc ? o : k(K_RLWINM, mask32(C, (w >> 1) & 31));
    // D-form loads and stores based on a register (rA = 0 is an address: step())
    case 32: return A ? k(K_LWZ, s16(w)) : o;
    case 34: return A ? k(K_LBZ, s16(w)) : o;
    case 40: return A ? k(K_LHZ, s16(w)) : o;
    case 42: return A ? k(K_LHA, s16(w)) : o;
    case 36: return A ? k(K_STW, s16(w)) : o;
    case 38: return A ? k(K_STB, s16(w)) : o;
    case 44: return A ? k(K_STH, s16(w)) : o;
    case 37: return k(K_STWU, s16(w));
    case 48: return A ? k(K_LFS, s16(w)) : o;
    case 50: return A ? k(K_LFD, s16(w)) : o;
    case 52: return A ? k(K_STFS, s16(w)) : o;
    case 54: return A ? k(K_STFD, s16(w)) : o;
    case 31: {
        const uint32_t xo = (w >> 1) & 0x3FF;
        switch (xo) {
        case 4: case 146: return k(K_TERM);                                 // tw, mtmsr
        case 0: o.cc = (uint8_t)((w >> 23) & 7); return k(K_CMPW);
        case 32: o.cc = (uint8_t)((w >> 23) & 7); return k(K_CMPLW);
        case 23: return k(K_LWZX);
        case 151: return k(K_STWX);
        case 87: return k(K_LBZX);
        case 215: return k(K_STBX);
        case 339: {
            const uint32_t n = A | B << 5;
            return n == 8 ? k(K_MFLR) : n == 9 ? k(K_MFCTR) : o;
        }
        case 467: {
            const uint32_t n = A | B << 5;
            return n == 8 ? k(K_MTLR) : n == 9 ? k(K_MTCTR) : o;
        }
        }
        if (rc) return o;
        switch (xo) {
        case 444: return k(K_OR);
        case 28: return k(K_AND);
        case 316: return k(K_XOR);
        case 266: return k(K_ADD);
        case 40: return k(K_SUBF);
        case 104: return k(K_NEG);
        case 235: return k(K_MULLW);
        case 24: return k(K_SLW);
        case 536: return k(K_SRW);
        case 824: return k(K_SRAWI);
        case 922: return k(K_EXTSH);
        case 954: return k(K_EXTSB);
        }
        return o;
    }
    case 59:
        if (rc) return o;
        switch ((w >> 1) & 31) {
        case 21: return k(K_FADDS);
        case 20: return k(K_FSUBS);
        case 25: return k(K_FMULS);
        case 29: return k(K_FMADDS);
        case 28: return k(K_FMSUBS);
        }
        return o;
    case 63: {
        const uint32_t xo = (w >> 1) & 0x3FF;
        if (xo == 0) { o.cc = (uint8_t)((w >> 23) & 7); return k(K_FCMPU); }
        if (rc) return o;
        switch (xo) {
        case 72: return k(K_FMR);
        case 40: return k(K_FNEG);
        case 12: return k(K_FRSP);
        }
        switch (xo & 31) {
        case 21: return xo == 21 ? k(K_FADD) : o;
        case 20: return xo == 20 ? k(K_FSUB) : o;
        case 25: return (xo & ~(31u << 5)) == 25 ? k(K_FMUL) : o;          // C is in the xo's top bits
        }
        return o;
    }
    }
    return o;
}

constexpr uint32_t kMaxOps = 32;

// Blocks by address (direct-mapped), in arenas per host thread. A block
// stays where it is while any interp_call of the thread may be running it:
// the arenas are only emptied from the outermost one, between blocks.
struct Blocks {
    static constexpr size_t kSlots = 16384, kArena = 8u << 20;
    Block* slot[kSlots] = {};
    std::vector<std::unique_ptr<uint8_t[]>> arenas;
    size_t used = kArena;
    int depth = 0;
    void* alloc(size_t n) {
        n = (n + 15) & ~size_t(15);
        if (used + n > kArena) {
            arenas.emplace_back(new uint8_t[kArena]);
            used = 0;
        }
        void* p = arenas.back().get() + used;
        used += n;
        return p;
    }
    void reset() {
        std::memset(slot, 0, sizeof slot);
        arenas.resize(1);
        used = 0;
    }
};

Blocks& blocks() {
    static thread_local std::unique_ptr<Blocks> b;
    if (!b) b.reset(new Blocks);
    return *b;
}

// Whether memory still holds a block's words (inline: most blocks are a few words)
inline bool same_words(const uint8_t* a, const uint8_t* b, uint32_t n) {
    uint32_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64_t x, y;
        std::memcpy(&x, a + 4 * i, 8);
        std::memcpy(&y, b + 4 * i, 8);
        if (x != y) return false;
    }
    if (i < n) {
        uint32_t x, y;
        std::memcpy(&x, a + 4 * i, 4);
        std::memcpy(&y, b + 4 * i, 4);
        if (x != y) return false;
    }
    return true;
}

// The block at pc, as memory holds it now
// What the block's fixed branch target resolves to (see Block::gen)
inline void resolve(Block* b, uint32_t target) {
    const uint32_t gen = ppc_lookup_generation();
    if (b->gen == gen) return;
    b->fn = ppc_lookup(target);
    b->to_interp = !b->fn && ppc_runtime_code(target);
    b->gen = gen;
}

const Block* block_at(Blocks& bs, const uint8_t* mem, uint32_t pc) {
    Block*& s = bs.slot[(pc >> 2) & (Blocks::kSlots - 1)];
    if (s && s->start == pc && same_words(mem + pc, s->words(), s->n)) return s;
    if (bs.depth == 1 && bs.arenas.size() > 4) bs.reset();          // nothing of them runs now
    Op ops[kMaxOps + 1];
    uint32_t n = 0;
    while (n < kMaxOps) {
        uint32_t raw;
        std::memcpy(&raw, mem + pc + 4 * n, 4);
        ops[n] = classify(PPC_BSWAP32(raw), pc + 4 * n);
        if (ends_block(ops[n++].kind)) break;
    }
    ops[n] = Op{K_END, 0, 0, 0, 0, 0, 0};
    Block* b = static_cast<Block*>(bs.alloc(offsetof(Block, ops) + sizeof(Op) * (n + 1) + 4 * n));
    b->start = pc;
    b->n = n;
    b->gen = 0;                                    // (generations start at 1)
    b->to_interp = false;
    b->fn = nullptr;
    std::memcpy(b->ops, ops, sizeof(Op) * (n + 1));
    std::memcpy(const_cast<uint8_t*>(b->words()), mem + pc, 4 * n);
    s = b;
    return b;
}

}  // namespace

void interp_call(PPCContext& c, uint32_t pc) {
    static void* const labels[K_COUNT] = {
        &&L_END, &&L_STEP, &&L_TERM,
        &&L_LI, &&L_ADDI, &&L_ORI, &&L_XORI, &&L_ANDI, &&L_CMPWI, &&L_CMPLWI, &&L_RLWINM,
        &&L_LWZ, &&L_LBZ, &&L_LHZ, &&L_LHA, &&L_STW, &&L_STB, &&L_STH, &&L_STWU,
        &&L_LFS, &&L_LFD, &&L_STFS, &&L_STFD,
        &&L_OR, &&L_AND, &&L_XOR, &&L_ADD, &&L_SUBF, &&L_NEG, &&L_MULLW, &&L_SLW, &&L_SRW, &&L_SRAWI,
        &&L_EXTSH, &&L_EXTSB,
        &&L_CMPW, &&L_CMPLW, &&L_LWZX, &&L_STWX, &&L_LBZX, &&L_STBX,
        &&L_MFLR, &&L_MTLR, &&L_MFCTR, &&L_MTCTR,
        &&L_FMR, &&L_FNEG, &&L_FADD, &&L_FSUB, &&L_FMUL, &&L_FADDS, &&L_FSUBS, &&L_FMULS, &&L_FMADDS,
        &&L_FMSUBS, &&L_FCMPU, &&L_FRSP,
        &&L_ADDIC,
        &&L_B, &&L_BL, &&L_BC, &&L_BLR,
    };
    // the code is in RAM (ppc_call_indirect checks), and g_mem does not move
    const uint8_t* const mem = g_mem;
    Blocks& bs = blocks();
    struct Depth {
        Blocks& bs;
        explicit Depth(Blocks& b) : bs(b) { ++bs.depth; }
        ~Depth() { --bs.depth; }
    } depth(bs);
#define NEXT do { ++o; goto *labels[o->kind]; } while (0)
    for (;;) {
        const Block* b = block_at(bs, mem, pc);
        const uint32_t start = b->start;
        const Op* o = b->ops;
        uint32_t ipc, target;
        goto *labels[o->kind];

    L_END:                                         // the block ran out without a branch
        pc = start + 4 * (uint32_t)(o - b->ops);
        continue;
    L_STEP: {
        const Step r = step(c, start + 4 * (uint32_t)(o - b->ops), o->w);
        if (PPC_UNLIKELY(r.flow != F_NEXT)) {      // (not for these kinds; as step() says anyway)
            ipc = start + 4 * (uint32_t)(o - b->ops);
            if (r.flow == F_RETURN) return;
            target = r.target;
            goto jump;
        }
        NEXT;
    }
    L_TERM: {
        // step() may run guest code (calls, syscalls, traps): nothing of this
        // block after it, which that code may have rewritten
        ipc = start + 4 * (uint32_t)(o - b->ops);
        const Step r = step(c, ipc, o->w);
        if (r.flow == F_NEXT) { pc = ipc + 4; continue; }
        if (r.flow == F_RETURN) return;
        target = r.target;
        goto jump;
    }

    L_LI: c.r[o->d] = o->imm; NEXT;
    L_ADDI: c.r[o->d] = c.r[o->a] + o->imm; NEXT;
    L_ORI: c.r[o->a] = c.r[o->d] | o->imm; NEXT;
    L_XORI: c.r[o->a] = c.r[o->d] ^ o->imm; NEXT;
    L_ANDI: c.r[o->a] = c.r[o->d] & o->imm; cr0_rc(c, c.r[o->a]); NEXT;
    L_CMPWI: cr_cmp_s(c, o->cc, (int32_t)c.r[o->a], (int32_t)o->imm); NEXT;
    L_CMPLWI: cr_cmp_u(c, o->cc, c.r[o->a], o->imm); NEXT;
    L_RLWINM: c.r[o->a] = rotl32(c.r[o->d], o->b) & o->imm; NEXT;

    L_LWZ: c.r[o->d] = ld32(c.r[o->a] + o->imm); NEXT;
    L_LBZ: c.r[o->d] = ld8(c.r[o->a] + o->imm); NEXT;
    L_LHZ: c.r[o->d] = ld16(c.r[o->a] + o->imm); NEXT;
    L_LHA: c.r[o->d] = (uint32_t)(int32_t)(int16_t)ld16(c.r[o->a] + o->imm); NEXT;
    L_STW: st32(c.r[o->a] + o->imm, c.r[o->d]); NEXT;
    L_STB: st8(c.r[o->a] + o->imm, (uint8_t)c.r[o->d]); NEXT;
    L_STH: st16(c.r[o->a] + o->imm, (uint16_t)c.r[o->d]); NEXT;
    L_STWU: {
        const uint32_t ea = c.r[o->a] + o->imm;
        st32(ea, c.r[o->d]);
        c.r[o->a] = ea;
        NEXT;
    }
    L_LFS: {
        const double v = (double)as_f32(ld32(c.r[o->a] + o->imm));
        c.f[o->d] = v;
        c.ps1[o->d] = v;
        NEXT;
    }
    L_LFD: c.f[o->d] = as_f64(ld64(c.r[o->a] + o->imm)); NEXT;
    L_STFS: st32(c.r[o->a] + o->imm, as_u32((float)c.f[o->d])); NEXT;
    L_STFD: st64(c.r[o->a] + o->imm, as_u64(c.f[o->d])); NEXT;

    L_OR: c.r[o->a] = c.r[o->d] | c.r[o->b]; NEXT;
    L_AND: c.r[o->a] = c.r[o->d] & c.r[o->b]; NEXT;
    L_XOR: c.r[o->a] = c.r[o->d] ^ c.r[o->b]; NEXT;
    L_ADD: c.r[o->d] = c.r[o->a] + c.r[o->b]; NEXT;
    L_SUBF: c.r[o->d] = c.r[o->b] - c.r[o->a]; NEXT;
    L_NEG: c.r[o->d] = 0u - c.r[o->a]; NEXT;
    L_MULLW: c.r[o->d] = (uint32_t)((int64_t)(int32_t)c.r[o->a] * (int32_t)c.r[o->b]); NEXT;
    L_SLW: c.r[o->a] = slw(c.r[o->d], c.r[o->b]); NEXT;
    L_SRW: c.r[o->a] = srw(c.r[o->d], c.r[o->b]); NEXT;
    L_SRAWI: c.r[o->a] = sraw(c, c.r[o->d], o->b); NEXT;
    L_EXTSH: c.r[o->a] = (uint32_t)(int32_t)(int16_t)c.r[o->d]; NEXT;
    L_EXTSB: c.r[o->a] = (uint32_t)(int32_t)(int8_t)c.r[o->d]; NEXT;
    L_CMPW: cr_cmp_s(c, o->cc, (int32_t)c.r[o->a], (int32_t)c.r[o->b]); NEXT;
    L_CMPLW: cr_cmp_u(c, o->cc, c.r[o->a], c.r[o->b]); NEXT;
    L_LWZX: c.r[o->d] = ld32((o->a ? c.r[o->a] : 0) + c.r[o->b]); NEXT;
    L_STWX: st32((o->a ? c.r[o->a] : 0) + c.r[o->b], c.r[o->d]); NEXT;
    L_LBZX: c.r[o->d] = ld8((o->a ? c.r[o->a] : 0) + c.r[o->b]); NEXT;
    L_STBX: st8((o->a ? c.r[o->a] : 0) + c.r[o->b], (uint8_t)c.r[o->d]); NEXT;
    L_MFLR: c.r[o->d] = c.lr; NEXT;
    L_MTLR: c.lr = c.r[o->d]; NEXT;
    L_MFCTR: c.r[o->d] = c.ctr; NEXT;
    L_MTCTR: c.ctr = c.r[o->d]; NEXT;

    L_FMR: c.f[o->d] = c.f[o->b]; NEXT;
    L_FNEG: c.f[o->d] = -c.f[o->b]; NEXT;
    L_FADD: c.f[o->d] = c.f[o->a] + c.f[o->b]; NEXT;
    L_FSUB: c.f[o->d] = c.f[o->a] - c.f[o->b]; NEXT;
    L_FMUL: c.f[o->d] = c.f[o->a] * c.f[o->cc]; NEXT;
    L_FADDS: fp_single(c, o->d, c.f[o->a] + c.f[o->b]); NEXT;
    L_FSUBS: fp_single(c, o->d, c.f[o->a] - c.f[o->b]); NEXT;
    L_FMULS: fp_single(c, o->d, c.f[o->a] * c.f[o->cc]); NEXT;
    L_FMADDS: fp_single(c, o->d, c.f[o->a] * c.f[o->cc] + c.f[o->b]); NEXT;
    L_FMSUBS: fp_single(c, o->d, c.f[o->a] * c.f[o->cc] - c.f[o->b]); NEXT;
    L_FCMPU: fcmp(c, o->cc, c.f[o->a], c.f[o->b]); NEXT;
    L_FRSP: fp_single(c, o->d, c.f[o->b]); NEXT;
    L_ADDIC: c.r[o->d] = add_ca(c, c.r[o->a], o->imm, 0); NEXT;

    L_B:
        ipc = start + 4 * (uint32_t)(o - b->ops);
        target = o->imm;
        goto jump_fixed;
    L_BL:                                          // a call, then on after it
        ipc = start + 4 * (uint32_t)(o - b->ops);
        c.lr = ipc + 4;
        if (o->imm != ipc + 4) {
            // as ppc_call_indirect, with its lookup kept in the block
            // (WIIKIT_ICALLS, which counts the calls, still goes through it)
            static const bool counting = std::getenv("WIIKIT_ICALLS") != nullptr;
            Block* mb = const_cast<Block*>(b);
            resolve(mb, o->imm);
            if (counting) ppc_call_indirect(c, o->imm);
            else if (mb->fn) mb->fn(c);
            else if (mb->to_interp) interp_call(c, o->imm);
            else ppc_call_indirect(c, o->imm);
        }
        pc = ipc + 4;
        continue;
    L_BC:
        ipc = start + 4 * (uint32_t)(o - b->ops);
        if (bc_taken(c, o->d, o->a)) {
            target = o->imm;
            goto jump_fixed;
        }
        pc = ipc + 4;
        continue;
    L_BLR:
        return;

    jump_fixed: {                                  // jump, with the lookup kept in the block
        Block* mb = const_cast<Block*>(b);
        resolve(mb, target);
        if (mb->fn) {
            mb->fn(c);
            return;
        }
        if (target <= ipc) PPC_POLL(c);
        pc = target;
        continue;
    }
    jump:
        if (PPCFunc f = ppc_lookup(target)) {      // a tail call into recompiled code
            f(c);
            return;
        }
        if (target <= ipc) PPC_POLL(c);            // a loop's back-edge
        pc = target;
    }
#undef NEXT
}
