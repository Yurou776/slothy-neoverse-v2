addvl x1, x0, #8

ld1h z0.h,p0/z,[x0]
ld1h z1.h,p0/z,[x0,#1,mul vl]
ld1h z2.h,p0/z,[x0,#2,mul vl]
ld1h z3.h,p0/z,[x0,#3,mul vl]
ld1h z4.h,p0/z,[x0,#4,mul vl]
ld1h z7.h,p0/z,[x0,#5,mul vl]
ld1h z16.h,p0/z,[x0,#6,mul vl]
ld1h z17.h,p0/z,[x0,#7,mul vl]

ld1h z18.h,p0/z,[x1]
ld1h z19.h,p0/z,[x1,#1,mul vl]
ld1h z20.h,p0/z,[x1,#2,mul vl]
ld1h z21.h,p0/z,[x1,#3,mul vl]
ld1h z22.h,p0/z,[x1,#4,mul vl]
ld1h z23.h,p0/z,[x1,#5,mul vl]
ld1h z24.h,p0/z,[x1,#6,mul vl]
ld1h z25.h,p0/z,[x1,#7,mul vl]

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z18
smulh z6.h, p3/m, z6.h, z28.h
mul z18.h, p3/m, z18.h, z29.h
smulh z18.h, p3/m, z18.h, z30.h
sub z6.h, z6.h, z18.h
sub z18.h, z0.h, z6.h
add z0.h, z0.h, z6.h

movprfx z6, z19
smulh z6.h, p3/m, z6.h, z28.h
mul z19.h, p3/m, z19.h, z29.h
smulh z19.h, p3/m, z19.h, z30.h
sub z6.h, z6.h, z19.h
sub z19.h, z1.h, z6.h
add z1.h, z1.h, z6.h

movprfx z6, z20
smulh z6.h, p3/m, z6.h, z28.h
mul z20.h, p3/m, z20.h, z29.h
smulh z20.h, p3/m, z20.h, z30.h
sub z6.h, z6.h, z20.h
sub z20.h, z2.h, z6.h
add z2.h, z2.h, z6.h

movprfx z6, z21
smulh z6.h, p3/m, z6.h, z28.h
mul z21.h, p3/m, z21.h, z29.h
smulh z21.h, p3/m, z21.h, z30.h
sub z6.h, z6.h, z21.h
sub z21.h, z3.h, z6.h
add z3.h, z3.h, z6.h

movprfx z6, z22
smulh z6.h, p3/m, z6.h, z28.h
mul z22.h, p3/m, z22.h, z29.h
smulh z22.h, p3/m, z22.h, z30.h
sub z6.h, z6.h, z22.h
sub z22.h, z4.h, z6.h
add z4.h, z4.h, z6.h

movprfx z6, z23
smulh z6.h, p3/m, z6.h, z28.h
mul z23.h, p3/m, z23.h, z29.h
smulh z23.h, p3/m, z23.h, z30.h
sub z6.h, z6.h, z23.h
sub z23.h, z7.h, z6.h
add z7.h, z7.h, z6.h

movprfx z6, z24
smulh z6.h, p3/m, z6.h, z28.h
mul z24.h, p3/m, z24.h, z29.h
smulh z24.h, p3/m, z24.h, z30.h
sub z6.h, z6.h, z24.h
sub z24.h, z16.h, z6.h
add z16.h, z16.h, z6.h

movprfx z6, z25
smulh z6.h, p3/m, z6.h, z28.h
mul z25.h, p3/m, z25.h, z29.h
smulh z25.h, p3/m, z25.h, z30.h
sub z6.h, z6.h, z25.h
sub z25.h, z17.h, z6.h
add z17.h, z17.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z4
smulh z6.h, p3/m, z6.h, z28.h
mul z4.h, p3/m, z4.h, z29.h
smulh z4.h, p3/m, z4.h, z30.h
sub z6.h, z6.h, z4.h
sub z4.h, z0.h, z6.h
add z0.h, z0.h, z6.h

movprfx z6, z7
smulh z6.h, p3/m, z6.h, z28.h
mul z7.h, p3/m, z7.h, z29.h
smulh z7.h, p3/m, z7.h, z30.h
sub z6.h, z6.h, z7.h
sub z7.h, z1.h, z6.h
add z1.h, z1.h, z6.h

movprfx z6, z16
smulh z6.h, p3/m, z6.h, z28.h
mul z16.h, p3/m, z16.h, z29.h
smulh z16.h, p3/m, z16.h, z30.h
sub z6.h, z6.h, z16.h
sub z16.h, z2.h, z6.h
add z2.h, z2.h, z6.h

movprfx z6, z17
smulh z6.h, p3/m, z6.h, z28.h
mul z17.h, p3/m, z17.h, z29.h
smulh z17.h, p3/m, z17.h, z30.h
sub z6.h, z6.h, z17.h
sub z17.h, z3.h, z6.h
add z3.h, z3.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z22
smulh z6.h, p3/m, z6.h, z28.h
mul z22.h, p3/m, z22.h, z29.h
smulh z22.h, p3/m, z22.h, z30.h
sub z6.h, z6.h, z22.h
sub z22.h, z18.h, z6.h
add z18.h, z18.h, z6.h

movprfx z6, z23
smulh z6.h, p3/m, z6.h, z28.h
mul z23.h, p3/m, z23.h, z29.h
smulh z23.h, p3/m, z23.h, z30.h
sub z6.h, z6.h, z23.h
sub z23.h, z19.h, z6.h
add z19.h, z19.h, z6.h

movprfx z6, z24
smulh z6.h, p3/m, z6.h, z28.h
mul z24.h, p3/m, z24.h, z29.h
smulh z24.h, p3/m, z24.h, z30.h
sub z6.h, z6.h, z24.h
sub z24.h, z20.h, z6.h
add z20.h, z20.h, z6.h

movprfx z6, z25
smulh z6.h, p3/m, z6.h, z28.h
mul z25.h, p3/m, z25.h, z29.h
smulh z25.h, p3/m, z25.h, z30.h
sub z6.h, z6.h, z25.h
sub z25.h, z21.h, z6.h
add z21.h, z21.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z2
smulh z6.h, p3/m, z6.h, z28.h
mul z2.h, p3/m, z2.h, z29.h
smulh z2.h, p3/m, z2.h, z30.h
sub z6.h, z6.h, z2.h
sub z2.h, z0.h, z6.h
add z0.h, z0.h, z6.h

movprfx z6, z3
smulh z6.h, p3/m, z6.h, z28.h
mul z3.h, p3/m, z3.h, z29.h
smulh z3.h, p3/m, z3.h, z30.h
sub z6.h, z6.h, z3.h
sub z3.h, z1.h, z6.h
add z1.h, z1.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z16
smulh z6.h, p3/m, z6.h, z28.h
mul z16.h, p3/m, z16.h, z29.h
smulh z16.h, p3/m, z16.h, z30.h
sub z6.h, z6.h, z16.h
sub z16.h, z4.h, z6.h
add z4.h, z4.h, z6.h

movprfx z6, z17
smulh z6.h, p3/m, z6.h, z28.h
mul z17.h, p3/m, z17.h, z29.h
smulh z17.h, p3/m, z17.h, z30.h
sub z6.h, z6.h, z17.h
sub z17.h, z7.h, z6.h
add z7.h, z7.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z20
smulh z6.h, p3/m, z6.h, z28.h
mul z20.h, p3/m, z20.h, z29.h
smulh z20.h, p3/m, z20.h, z30.h
sub z6.h, z6.h, z20.h
sub z20.h, z18.h, z6.h
add z18.h, z18.h, z6.h

movprfx z6, z21
smulh z6.h, p3/m, z6.h, z28.h
mul z21.h, p3/m, z21.h, z29.h
smulh z21.h, p3/m, z21.h, z30.h
sub z6.h, z6.h, z21.h
sub z21.h, z19.h, z6.h
add z19.h, z19.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z24
smulh z6.h, p3/m, z6.h, z28.h
mul z24.h, p3/m, z24.h, z29.h
smulh z24.h, p3/m, z24.h, z30.h
sub z6.h, z6.h, z24.h
sub z24.h, z22.h, z6.h
add z22.h, z22.h, z6.h

movprfx z6, z25
smulh z6.h, p3/m, z6.h, z28.h
mul z25.h, p3/m, z25.h, z29.h
smulh z25.h, p3/m, z25.h, z30.h
sub z6.h, z6.h, z25.h
sub z25.h, z23.h, z6.h
add z23.h, z23.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z1
smulh z6.h, p3/m, z6.h, z28.h
mul z1.h, p3/m, z1.h, z29.h
smulh z1.h, p3/m, z1.h, z30.h
sub z6.h, z6.h, z1.h
sub z1.h, z0.h, z6.h
add z0.h, z0.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z3
smulh z6.h, p3/m, z6.h, z28.h
mul z3.h, p3/m, z3.h, z29.h
smulh z3.h, p3/m, z3.h, z30.h
sub z6.h, z6.h, z3.h
sub z3.h, z2.h, z6.h
add z2.h, z2.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z7
smulh z6.h, p3/m, z6.h, z28.h
mul z7.h, p3/m, z7.h, z29.h
smulh z7.h, p3/m, z7.h, z30.h
sub z6.h, z6.h, z7.h
sub z7.h, z4.h, z6.h
add z4.h, z4.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z17
smulh z6.h, p3/m, z6.h, z28.h
mul z17.h, p3/m, z17.h, z29.h
smulh z17.h, p3/m, z17.h, z30.h
sub z6.h, z6.h, z17.h
sub z17.h, z16.h, z6.h
add z16.h, z16.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z19
smulh z6.h, p3/m, z6.h, z28.h
mul z19.h, p3/m, z19.h, z29.h
smulh z19.h, p3/m, z19.h, z30.h
sub z6.h, z6.h, z19.h
sub z19.h, z18.h, z6.h
add z18.h, z18.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z21
smulh z6.h, p3/m, z6.h, z28.h
mul z21.h, p3/m, z21.h, z29.h
smulh z21.h, p3/m, z21.h, z30.h
sub z6.h, z6.h, z21.h
sub z21.h, z20.h, z6.h
add z20.h, z20.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z23
smulh z6.h, p3/m, z6.h, z28.h
mul z23.h, p3/m, z23.h, z29.h
smulh z23.h, p3/m, z23.h, z30.h
sub z6.h, z6.h, z23.h
add z22.h, z22.h, z6.h
sub z23.h, z22.h, z6.h

ld1rh {z28.h}, p3/z, [x16]
ld1rh {z29.h}, p3/z, [x17]
add x16, x16, #2
add x17, x17, #2

movprfx z6, z25
smulh z6.h, p3/m, z6.h, z28.h
mul z25.h, p3/m, z25.h, z29.h
smulh z25.h, p3/m, z25.h, z30.h
sub z6.h, z6.h, z25.h
sub z25.h, z24.h, z6.h
add z24.h, z24.h, z6.h

st1h z0.h,p0,[x0]
st1h z1.h,p0,[x0,#1,mul vl]
st1h z2.h,p0,[x0,#2,mul vl]
st1h z3.h,p0,[x0,#3,mul vl]
st1h z4.h,p0,[x0,#4,mul vl]
st1h z7.h,p0,[x0,#5,mul vl]
st1h z16.h,p0,[x0,#6,mul vl]
st1h z17.h,p0,[x0,#7,mul vl]
st1h z18.h,p0,[x1]
st1h z19.h,p0,[x1,#1,mul vl]
st1h z20.h,p0,[x1,#2,mul vl]
st1h z21.h,p0,[x1,#3,mul vl]
st1h z22.h,p0,[x1,#4,mul vl]
st1h z23.h,p0,[x1,#5,mul vl]
st1h z24.h,p0,[x1,#6,mul vl]
st1h z25.h,p0,[x1,#7,mul vl]