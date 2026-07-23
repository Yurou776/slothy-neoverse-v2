slothy_start:
        ldp x3, x4, [x0]
        ldp x5, x6, [x0, #16]
        ldr x7, [x1]
        ldr x8, [x1, #8]

        adds x9,  x3, x4
        adcs x10, x5, x6
        adc  x11, x7, xzr
        subs x12, x3, x5
        sbcs x13, x4, x6

        and  x14, x9, x10
        ands x15, x11, #0xff
        csel x29, x13, x14, ne
        eor  x16, x13, x14
        orr  x17, x15, x16
        bic  x18, x16, x17

        mul   x19, x3, x4
        umulh x20, x3, x4
        madd  x21, x5, x6, x19
        msub  x22, x5, x6, x20
        mneg  x23, x8, x8

        lsl  x24, x19, #3
        lsr  x25, x20, #7
        ror  x26, x21, #13
        extr x27, x22, x23, #5
        bfi  x28, x24, #8, #16

        cmp  x9, x10
        ccmp x11, x12, #0, eq
        cinc x30, x15, eq

        tst  x16, #0xff
        cset x1, ne

        movz x2, #0x1234
        movk x2, #0xabcd, lsl #16
        add  x2, x2, x29, lsl #2
        eor  x2, x2, x30, lsr #4

        stp x9,  x10, [x0]
        stp x19, x20, [x0, #16]
        stp x14, x18, [x0, #32]
        stp x25, x26, [x0, #48]
        stp x27, x28, [x0, #64]
        stp x21, x22, [x0, #80]
        stp x23, x24, [x0, #96]
        stp x12, x17, [x0, #112]
        stp x2,  x1,  [x0, #128]
slothy_end:
