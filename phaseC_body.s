ld1h z0.h,p0/z,[x1]
ld1h z1.h,p0/z,[x1,#1,mul vl]
ld1h z2.h,p0/z,[x1,#2,mul vl]
ld1h z3.h,p0/z,[x1,#3,mul vl]

uzp1 z4.s,z0.s,z1.s
uzp2 z5.s,z0.s,z1.s
uzp1 z16.s,z2.s,z3.s
uzp2 z17.s,z2.s,z3.s

ld1h z28.h,p3/z,[x2]
ld1h z29.h,p3/z,[x3]

movprfx z6,z5
smulh z6.h,p3/m,z6.h,z28.h
mul z5.h,p3/m,z5.h,z29.h
smulh z5.h,p3/m,z5.h,z30.h
sub z6.h,z6.h,z5.h
sub z5.h,z4.h,z6.h
add z4.h,z4.h,z6.h

ld1h z28.h,p3/z,[x2,#1,mul vl]
ld1h z29.h,p3/z,[x3,#1,mul vl]

movprfx z6,z17
smulh z6.h,p3/m,z6.h,z28.h
mul z17.h,p3/m,z17.h,z29.h
smulh z17.h,p3/m,z17.h,z30.h
sub z6.h,z6.h,z17.h
sub z17.h,z16.h,z6.h
add z16.h,z16.h,z6.h

zip1 z0.s,z4.s,z5.s
zip2 z1.s,z4.s,z5.s
zip1 z2.s,z16.s,z17.s
zip2 z3.s,z16.s,z17.s

st1h z0.h,p0,[x1]
st1h z1.h,p0,[x1,#1,mul vl]
st1h z2.h,p0,[x1,#2,mul vl]
st1h z3.h,p0,[x1,#3,mul vl]
