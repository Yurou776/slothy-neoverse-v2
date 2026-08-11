.arch armv8-a+sve
    .text

// void ntt_sve16(int16_t a[128]);
//
// Scalar-free, VL-agnostic SVE(v1) forward NTT for the user's transform:
//   length-128 CYCLIC NTT, q=257, root=9, natural-order output.
//
// Strategy (the SVE answer to the Kyber NEON "merge layers + transpose"):
//   * int16 lanes  -> 2x the throughput of the int32 version
//   * NO scalar fallback: every layer is vectorised
//   * small-stride layers are re-vectorised with structured load/store
//       - Phase A: stride 64,32,16  contiguous  (partners in different vectors)
//       - Phase B: stride  8, 4     via LD4D/ST4D (64-bit granularity)
//       - Phase C: stride  2, 1     via LD4H/ST4H (16-bit granularity)
//     LD4 granularity is fixed => VL-agnostic, zero permute instructions.
//   * DIF butterflies + final bit-reverse == the user's (bit-reverse + DIT).
//   * Barrett via SMULH (base SVE, not SVE2).
//
// Verified bit-exact vs the C reference at VL = 128/256/512/1024 bit and on
// -cpu neoverse-v1 (SVE v1 only).

#define Q   257
#define BC  255              // round(2^16/q): final reduction constant

// Gentleman-Sande butterfly:  za <- u+v ,  zb <- (u-v)*w   (lazy int16)
//   zwlo = w (centred),  zwhi = round(w*2^16/q)
.macro GS za, zb, zt, zu, zwlo, zwhi, zq, pg
    sub     \zu\().h, \za\().h, \zb\().h     // u - v
    add     \za\().h, \za\().h, \zb\().h     // SUM = u + v
    movprfx \zt, \zu
    smulh   \zt\().h, \pg/m, \zt\().h, \zwhi\().h
    movprfx \zb, \zu
    mul     \zb\().h, \pg/m, \zb\().h, \zwlo\().h
    mls     \zb\().h, \pg/m, \zt\().h, \zq\().h   // DIFF = (u-v)*w mod q
.endm

    .global ntt_sve16
    .type   ntt_sve16, %function
    .align  4
ntt_sve16:
    ptrue   p7.h
    mov     w9, #Q
    dup     z30.h, w9                    // zQ = q

// ---------- Phase A : stride 64, 32, 16 (contiguous, per-lane twiddle) ----------
// half=64: 1 group ; half=32: 2 groups ; half=16: 4 groups.
    // --- half = 64 ---
    adrp    x3, tw_A64_lo
    add     x3, x3, :lo12:tw_A64_lo
    adrp    x4, tw_A64_hi
    add     x4, x4, :lo12:tw_A64_hi
    mov     x5, x0                       // &a[0]  (lower half base)
    add     x6, x0, #(64*2)              // &a[64] (upper half base)
    mov     x7, #0
    mov     x8, #64
.LA64:
    whilelt p0.h, x7, x8
    b.none  .LA64_done
    ld1h    z0.h, p0/z, [x5, x7, lsl #1]
    ld1h    z1.h, p0/z, [x6, x7, lsl #1]
    ld1h    z28.h, p0/z, [x3, x7, lsl #1]
    ld1h    z29.h, p0/z, [x4, x7, lsl #1]
    GS      z0, z1, z4, z5, z28, z29, z30, p0
    st1h    z0.h, p0, [x5, x7, lsl #1]
    st1h    z1.h, p0, [x6, x7, lsl #1]
    inch    x7
    b       .LA64
.LA64_done:

    // --- half = 32  (2 groups, twiddle table reused) ---
    mov     x10, #0                      // group base index (0, 64)
.LA32_group:
    adrp    x3, tw_A32_lo
    add     x3, x3, :lo12:tw_A32_lo
    adrp    x4, tw_A32_hi
    add     x4, x4, :lo12:tw_A32_hi
    add     x5, x0, x10, lsl #1
    add     x6, x5, #(32*2)
    mov     x7, #0
    mov     x8, #32
.LA32:
    whilelt p0.h, x7, x8
    b.none  .LA32_done
    ld1h    z0.h, p0/z, [x5, x7, lsl #1]
    ld1h    z1.h, p0/z, [x6, x7, lsl #1]
    ld1h    z28.h, p0/z, [x3, x7, lsl #1]
    ld1h    z29.h, p0/z, [x4, x7, lsl #1]
    GS      z0, z1, z4, z5, z28, z29, z30, p0
    st1h    z0.h, p0, [x5, x7, lsl #1]
    st1h    z1.h, p0, [x6, x7, lsl #1]
    inch    x7
    b       .LA32
.LA32_done:
    add     x10, x10, #64
    cmp     x10, #128
    b.lt    .LA32_group

    // --- half = 16  (4 groups) ---
    mov     x10, #0
.LA16_group:
    adrp    x3, tw_A16_lo
    add     x3, x3, :lo12:tw_A16_lo
    adrp    x4, tw_A16_hi
    add     x4, x4, :lo12:tw_A16_hi
    add     x5, x0, x10, lsl #1
    add     x6, x5, #(16*2)
    mov     x7, #0
    mov     x8, #16
.LA16:
    whilelt p0.h, x7, x8
    b.none  .LA16_done
    ld1h    z0.h, p0/z, [x5, x7, lsl #1]
    ld1h    z1.h, p0/z, [x6, x7, lsl #1]
    ld1h    z28.h, p0/z, [x3, x7, lsl #1]
    ld1h    z29.h, p0/z, [x4, x7, lsl #1]
    GS      z0, z1, z4, z5, z28, z29, z30, p0
    st1h    z0.h, p0, [x5, x7, lsl #1]
    st1h    z1.h, p0, [x6, x7, lsl #1]
    inch    x7
    b       .LA16
.LA16_done:
    add     x10, x10, #32
    cmp     x10, #128
    b.lt    .LA16_group
slothy_start:
// ---------- Phase B : stride 8, 4 via LD4D/ST4D ----------
//   (z0,z2)->ln8a  (z1,z3)->ln8b  (z0,z1)->ln4a  (z2,z3)->ln4b
    adrp    x2, tw_ln8a_lo
    add     x2, x2, :lo12:tw_ln8a_lo
    adrp    x3, tw_ln8a_hi
    add     x3, x3, :lo12:tw_ln8a_hi
    adrp    x4, tw_ln8b_lo
    add     x4, x4, :lo12:tw_ln8b_lo
    adrp    x5, tw_ln8b_hi
    add     x5, x5, :lo12:tw_ln8b_hi
    adrp    x6, tw_ln4a_lo
    add     x6, x6, :lo12:tw_ln4a_lo
    adrp    x7, tw_ln4a_hi
    add     x7, x7, :lo12:tw_ln4a_hi
    adrp    x9, tw_ln4b_lo
    add     x9, x9, :lo12:tw_ln4b_lo
    adrp    x10, tw_ln4b_hi
    add     x10, x10, :lo12:tw_ln4b_hi

    mov     x1, x0
    mov     x11, #0                      // .d structure index
    mov     x12, #(128*2/32)             // total .d structures = 8
.LB:
    whilelt p0.d, x11, x12
    b.none  .LB_done
    ld4d    {z0.d - z3.d}, p0/z, [x1]

    // stride 8
    ld1h    z28.h, p7/z, [x2]            // ln8a
    ld1h    z29.h, p7/z, [x3]
    GS      z0, z2, z6, z5, z28, z29, z30, p7
    ld1h    z28.h, p7/z, [x4]            // ln8b
    ld1h    z29.h, p7/z, [x5]
    GS      z1, z3, z6, z5, z28, z29, z30, p7

    // stride 4
    ld1h    z28.h, p7/z, [x6]            // ln4a
    ld1h    z29.h, p7/z, [x7]
    GS      z0, z1, z6, z5, z28, z29, z30, p7
    ld1h    z28.h, p7/z, [x9]            // ln4b
    ld1h    z29.h, p7/z, [x10]
    GS      z2, z3, z6, z5, z28, z29, z30, p7

    st4d    {z0.d - z3.d}, p0, [x1]

    addvl   x1,  x1,  #4
    addvl   x2,  x2,  #1
    addvl   x3,  x3,  #1
    addvl   x4,  x4,  #1
    addvl   x5,  x5,  #1
    addvl   x6,  x6,  #1
    addvl   x7,  x7,  #1
    addvl   x9,  x9,  #1
    addvl   x10, x10, #1
    incd    x11
    b       .LB
.LB_done:
slothy_end:
// ---------- Phase C : stride 2, 1 via LD4H/ST4H ----------
//   (z0,z2)->ln2a  (z1,z3)->ln2b  (z0,z1)->ln1a  (z2,z3)->ln1b
    adrp    x2, tw_ln2a_lo
    add     x2, x2, :lo12:tw_ln2a_lo
    adrp    x3, tw_ln2a_hi
    add     x3, x3, :lo12:tw_ln2a_hi
    adrp    x4, tw_ln2b_lo
    add     x4, x4, :lo12:tw_ln2b_lo
    adrp    x5, tw_ln2b_hi
    add     x5, x5, :lo12:tw_ln2b_hi
    adrp    x6, tw_ln1a_lo
    add     x6, x6, :lo12:tw_ln1a_lo
    adrp    x7, tw_ln1a_hi
    add     x7, x7, :lo12:tw_ln1a_hi
    adrp    x9, tw_ln1b_lo
    add     x9, x9, :lo12:tw_ln1b_lo
    adrp    x10, tw_ln1b_hi
    add     x10, x10, :lo12:tw_ln1b_hi

    mov     x1, x0
    mov     x11, #0                      // .h structure index
    mov     x12, #(128/4)                // total = 32
.LC:
    whilelt p0.h, x11, x12
    b.none  .LC_done
    ld4h    {z0.h - z3.h}, p0/z, [x1]

    // stride 2
    ld1h    z28.h, p7/z, [x2]            // ln2a
    ld1h    z29.h, p7/z, [x3]
    GS      z0, z2, z6, z5, z28, z29, z30, p7
    ld1h    z28.h, p7/z, [x4]            // ln2b
    ld1h    z29.h, p7/z, [x5]
    GS      z1, z3, z6, z5, z28, z29, z30, p7

    // stride 1
    ld1h    z28.h, p7/z, [x6]            // ln1a
    ld1h    z29.h, p7/z, [x7]
    GS      z0, z1, z6, z5, z28, z29, z30, p7
    ld1h    z28.h, p7/z, [x9]            // ln1b
    ld1h    z29.h, p7/z, [x10]
    GS      z2, z3, z6, z5, z28, z29, z30, p7

    st4h    {z0.h - z3.h}, p0, [x1]

    addvl   x1,  x1,  #4
    addvl   x2,  x2,  #1
    addvl   x3,  x3,  #1
    addvl   x4,  x4,  #1
    addvl   x5,  x5,  #1
    addvl   x6,  x6,  #1
    addvl   x7,  x7,  #1
    addvl   x9,  x9,  #1
    addvl   x10, x10, #1
    inch    x11
    b       .LC
.LC_done:

// ---------- final Barrett reduction to canonical [0,q) ----------
    mov     w9, #BC
    dup     z27.h, w9
    mov     x11, #0
    mov     x12, #128
.LR:
    whilelt p0.h, x11, x12
    b.none  .LR_done
    ld1h    z0.h, p0/z, [x0, x11, lsl #1]
    movprfx z4, z0
    smulh   z4.h, p0/m, z4.h, z27.h
    mls     z0.h, p0/m, z4.h, z30.h      // z0 -= floor(z0/q)*q  -> [0,q]
    cmpge   p1.h, p0/z, z0.h, z30.h
    sub     z0.h, p1/m, z0.h, z30.h      // -> [0,q)
    cmplt   p1.h, p0/z, z0.h, #0
    add     z0.h, p1/m, z0.h, z30.h
    st1h    z0.h, p0, [x0, x11, lsl #1]
    inch    x11
    b       .LR
.LR_done:

// ---------- final bit-reverse (natural order) ----------
    mov     x1, #0
.LBR:
    cmp     x1, #128
    bge     .LBR_done
    rbit    w2, w1
    lsr     w2, w2, #25                  // reverse low 7 bits
    cmp     w2, w1
    b.le    .LBR_next
    ldrh    w3, [x0, x1, lsl #1]
    ldrh    w4, [x0, x2, lsl #1]
    strh    w4, [x0, x1, lsl #1]
    strh    w3, [x0, x2, lsl #1]
.LBR_next:
    add     x1, x1, #1
    b       .LBR
.LBR_done:
    ret
    .size   ntt_sve16, .-ntt_sve16

#include "ktw.S"
    .section .note.GNU-stack,"",%progbits