slothy_start:
        ldr q0, [x0]
        ldr q1, [x0, #16]
        ldr q2, [x1]
        ldr q3, [x1, #16]

        pmull  v4.1q, v0.1d, v2.1d
        pmull2 v5.1q, v0.2d, v2.2d
        pmull  v6.1q, v1.1d, v3.1d
        pmull2 v7.1q, v1.2d, v3.2d

        eor v8.16b,  v4.16b, v5.16b
        eor v9.16b,  v6.16b, v7.16b
        eor v10.16b, v8.16b, v9.16b
        eor v11.16b, v4.16b, v10.16b

        ext v12.16b, v11.16b, v11.16b, #8
        eor v13.16b, v11.16b, v12.16b

        str q13, [x2]
        str q10, [x2, #16]
slothy_end:
