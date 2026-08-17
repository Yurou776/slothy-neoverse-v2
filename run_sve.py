from slothy import Slothy
import slothy.targets.aarch64.aarch64_sve as arch
import slothy.targets.aarch64.neoverse_v1 as target

slothy = Slothy(arch, target)

slothy.load_source_from_file("phaseA.s")

slothy.config.variable_size = True
slothy.config.selftest = False

slothy.config.outputs = [
    "z0", "z1", "z2", "z3",
    "z4", "z7", "z16", "z17",
    "z18", "z19", "z20", "z21",
    "z22", "z23", "z24", "z25",
    "x16", "x17",
]

slothy.config.locked_registers = {
    "x0",
    "x1",
    "x16",
    "x17",

    "p0",
    "p3",

    "z0", "z1", "z2", "z3",
    "z4", "z7",
    "z16", "z17",
    "z18", "z19", "z20", "z21",
    "z22", "z23", "z24", "z25",

    "z28",
    "z29",
    "z30",
}

slothy.optimize()

slothy.write_source_to_file("phaseA_opt.s")