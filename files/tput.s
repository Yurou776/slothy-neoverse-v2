slothy_start:
        eor v0.16b, v16.16b, v17.16b
        eor v1.16b, v17.16b, v18.16b
        eor v2.16b, v18.16b, v19.16b
        eor v3.16b, v19.16b, v20.16b
        eor v4.16b, v20.16b, v21.16b
        eor v5.16b, v21.16b, v22.16b
        eor v6.16b, v22.16b, v23.16b
        eor v7.16b, v23.16b, v24.16b
        eor v8.16b, v24.16b, v25.16b
        eor v9.16b, v25.16b, v26.16b
        eor v10.16b, v26.16b, v27.16b
        eor v11.16b, v27.16b, v28.16b
        eor v12.16b, v28.16b, v29.16b
        eor v13.16b, v29.16b, v30.16b
        eor v14.16b, v30.16b, v31.16b
        eor v15.16b, v31.16b, v16.16b
        stp q0, q1, [x0, #0]
        stp q2, q3, [x0, #32]
        stp q4, q5, [x0, #64]
        stp q6, q7, [x0, #96]
        stp q8, q9, [x0, #128]
        stp q10, q11, [x0, #160]
        stp q12, q13, [x0, #192]
        stp q14, q15, [x0, #224]
slothy_end:
