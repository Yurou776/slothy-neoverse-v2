#
# Copyright (c) 2022 Arm Limited
# Copyright (c) 2022 Hanno Becker
# Copyright (c) 2023 Amin Abdulrahman, Matthias Kannwischer
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
SLOTHY microarchitecture model for the Arm Neoverse V2 core.

All latency / throughput / pipeline data is taken from:

    Arm(R) Neoverse(TM) V2 Core Software Optimization Guide,
    Issue 3.0 (r0p2), PJDOC-466751330-593177.

Section numbers in the comments below refer to that document.

Conventions used when transcribing the SWOG tables
--------------------------------------------------
* "Utilized Pipelines" maps directly onto the ``ExecutionUnit`` classmethods,
  which mirror the legend in Table 3-1 (plus ``V23``, see the note there).
* SLOTHY wants *inverse* throughput (cycles a unit is held), while the SWOG
  gives throughput (instructions per cycle).  Most of the SWOG throughput
  figure is already expressed by the number of pipelines in the symbol
  (e.g. ``I`` has 6 pipes and tput 6), in which case the inverse throughput
  here is simply 1.  It is only >1 where the SWOG throughput is *lower* than
  the pipeline count, using ``inverse_throughput = round(pipes / tput)``.

.. warning::

    Consider disabling the EOR3 fusion callback on this target.
    ``aarch64_neon.py`` installs ``veor.global_fusion_cb = eor3_fusion_cb()``,
    which merges two EORs into one EOR3.  On the V2, EOR3 is a Crypto SHA3 op
    (Table 3-21): pipeline ``V0``, throughput **1**, while a plain EOR is an
    ASIMD logical op on ``V`` with throughput 4.  Fusing therefore trades
    4 EOR/cycle for 1 EOR3/cycle -- a net loss for EOR-heavy code such as
    GF(2) arithmetic.
"""

from enum import Enum

from slothy.helper import lookup_multidict
from slothy.targets.aarch64.aarch64_neon import (
    find_class,
    all_subclass_leaves,
    # --- scalar loads -------------------------------------------------------
    Ldr_X,
    Ldp_X,
    Ldp_W,
    ldr_const,
    ldr_sxtw_wform,
    # --- scalar stores ------------------------------------------------------
    Str_X,
    Stp_X,
    Stp_W,
    w_stp_with_imm_sp,
    # --- vector loads -------------------------------------------------------
    Ldr_D,
    Ldr_Q,
    Ldp_Q,
    Ld2,
    Ld3,
    Ld4,
    q_ld1_2,
    q_ld2_lane_s,
    Q_Ld2_Lane_Post_Inc,
    q_ldr1_stack,
    q_ldr1_post_inc,
    b_ldr_stack_with_inc,
    d_ldr_stack_with_inc,
    # --- vector stores ------------------------------------------------------
    Str_Q,
    Stp_Q,
    St2,
    St3,
    St4,
    q_st1_4_with_postinc,
    d_str_stack_with_inc,
    d_stp_stack_with_inc,
    q_stp_stack_with_inc,
    # --- scalar ALU ---------------------------------------------------------
    AArch64BasicArithmetic,
    AArch64Logical,
    AArch64LogicalShifted,
    AArch64ShiftedArithmetic,
    AArch64ConditionalSelect,
    AArch64ConditionalCompare,
    AArch64Move,
    AArch64Shift,
    Tst,
    fcsel,
    bfi,
    extr,
    mov,
    movk_imm,
    movk_imm_lsl,
    lsr,
    lsr_wform,
    asr_wform,
    and_imm_wform,
    nop,
    # --- scalar multiply / CRC ---------------------------------------------
    AArch64HighMultiply,
    mul_xform,
    mneg_xform,
    madd_xform,
    msub_xform,
    mul_wform,
    umull_wform,
    umaddl_wform,
    AArch64CRC32,
    # --- FP transfer --------------------------------------------------------
    fmov_s_form,
    fmov_d_form,
    fmov_0,
    fmov_0_force_output,
    fmov_1,
    fmov_1_force_output,
    # --- ASIMD arithmetic / logic ------------------------------------------
    vadd,
    vsub,
    uaddlp,
    ASimdCompare,
    AArch64NeonLogical,
    AArch64NeonCount,
    vbif,
    vuaddlv_sform,
    # --- ASIMD permute / misc ----------------------------------------------
    Vzip,
    Vrev,
    Transpose,
    vext,
    vxtn,
    vmov,
    vmovi,
    vtbl,
    vtbl_2,
    vdup,
    vdup_w,
    mov_b00,
    mov_d01,
    mov_vtov_d,
    vmov_d,
    VecToGprMov,
    Vins,
    Mov_xtov_d,
    mov_wtov_s,
    # --- ASIMD multiply -----------------------------------------------------
    Vmul,
    Vmla,
    Vqdmulh,
    Vmull,
    Vmlal,
    # --- ASIMD shift --------------------------------------------------------
    VShiftImmediateBasic,
    VShiftImmediateRounding,
    VShiftRegBasic,
    AArch64NeonShiftInsert,
    vusra,
    # --- crypto -------------------------------------------------------------
    AESInstruction,
    SHA3Instruction,
    aesr_x2,
    aesr_x4,
    aese_x4,
    # --- virtual spill instructions ----------------------------------------
    qsave,
    qrestore,
    save,
    restore,
)

# Section 4.1: the dispatch stage can process up to 8 MOPs per cycle.
issue_rate = 8
llvm_mca_target = "neoverse-v2"


class ExecutionUnit(Enum):
    """Execution units of the Neoverse V2 core.

    The 17 issue pipelines are listed in Figure 2-1 / Table 2-1.  The
    classmethods below reproduce the pipeline symbols of Table 3-1, one
    method per symbol.

    .. note::

        ``V23`` is *not* in the Table 3-1 legend, but is used by Table 3-15
        for the (8x8) polynomial multiplies (PMUL / PMULL).  The legend in
        the SWOG is incomplete.
    """

    BR0 = 0
    BR1 = 1

    INT0 = 2  # Integer Single-Cycle 0..3
    INT1 = 3
    INT2 = 4
    INT3 = 5

    MUL0 = 6  # Integer Single/Multi-Cycle 0..1
    MUL1 = 7

    VEC0 = 8  # FP/ASIMD 0..3
    VEC1 = 9
    VEC2 = 10
    VEC3 = 11

    LS0 = 12  # Load/Store 0..1
    LS1 = 13
    LOAD2 = 14  # Load 2 (load only)

    STORE0 = 15  # Store data 0..1
    STORE1 = 16

    def __repr__(self):
        return self.name

    @classmethod
    def B(cls):  # noqa: E743
        """Branch 0/1"""
        return [cls.BR0, cls.BR1]

    @classmethod
    def R(cls):
        """Integer single cycle 0/1"""
        return [cls.INT0, cls.INT1]

    @classmethod
    def S(cls):
        """Integer single cycle 0/1/2/3"""
        return [cls.INT0, cls.INT1, cls.INT2, cls.INT3]

    @classmethod
    def M(cls):
        """Integer single/multicycle 0/1"""
        return [cls.MUL0, cls.MUL1]

    @classmethod
    def M0(cls):
        """Integer multicycle 0"""
        return [cls.MUL0]

    @classmethod
    def F(cls):
        """Integer single cycle 0/1 and single/multicycle 0/1"""
        return cls.R() + cls.M()

    @classmethod
    def I(cls):  # noqa: E743
        """Integer single cycle 0/1/2/3 and single/multicycle 0/1"""
        return cls.S() + cls.M()

    @classmethod
    def L01(cls):
        """Load/Store 0/1"""
        return [cls.LS0, cls.LS1]

    @classmethod
    def L(cls):
        """Load/Store 0/1 and Load 2"""
        return cls.L01() + [cls.LOAD2]

    @classmethod
    def D(cls):
        """Store data 0/1"""
        return [cls.STORE0, cls.STORE1]

    @classmethod
    def V(cls):
        """FP/ASIMD 0/1/2/3"""
        return [cls.VEC0, cls.VEC1, cls.VEC2, cls.VEC3]

    @classmethod
    def V01(cls):
        """FP/ASIMD 0/1"""
        return [cls.VEC0, cls.VEC1]

    @classmethod
    def V02(cls):
        """FP/ASIMD 0/2"""
        return [cls.VEC0, cls.VEC2]

    @classmethod
    def V13(cls):
        """FP/ASIMD 1/3"""
        return [cls.VEC1, cls.VEC3]

    @classmethod
    def V23(cls):
        """FP/ASIMD 2/3 (used by Table 3-15, missing from the Table 3-1 legend)"""
        return [cls.VEC2, cls.VEC3]

    @classmethod
    def V0(cls):
        """FP/ASIMD 0"""
        return [cls.VEC0]

    @classmethod
    def V1(cls):
        """FP/ASIMD 1"""
        return [cls.VEC1]


###############################################################################
#                                                                             #
# Multi-pipeline combinations                                                 #
#                                                                             #
# A list-of-lists value means "pick one of the outer entries, and occupy      #
# *all* units in it simultaneously".                                          #
#                                                                             #
###############################################################################

# Table 3-8: GPR stores use an address uOP (L01) and a store-data uOP (D).
_GPR_STORE = [[ls, d] for ls in ExecutionUnit.L01() for d in ExecutionUnit.D()]

# Table 3-14 / 3-20: vector stores take their data from the V01 pipes,
# not from the store-data pipes.
_VEC_STORE = [[ls, v] for ls in ExecutionUnit.L01() for v in ExecutionUnit.V01()]

# Table 3-18 "ASIMD transfer, gen reg to element": M0, V
_GPR_TO_ELEMENT = [[ExecutionUnit.MUL0, v] for v in ExecutionUnit.V()]

# Table 3-12 "FP transfer, from gen to high half of vec reg": M0, V
_GPR_TO_HIGH_HALF = _GPR_TO_ELEMENT


###############################################################################
#                                                                             #
# Instance-level predicates                                                   #
#                                                                             #
# Some SLOTHY instruction classes lump together SWOG rows with different       #
# characteristics (most notably flag-setting vs. non-flag-setting ALU ops).    #
# lookup_multidict accepts callables as keys, so they are split here.          #
#                                                                             #
# IMPORTANT: whenever a callable below narrows a class, that class must NOT    #
# also appear as a plain key, otherwise lookup_multidict raises                #
# "Multiple matches found".                                                    #
#                                                                             #
###############################################################################


def _sets_flags(inst):
    """True if the instruction writes NZCV (modifiesFlags=True at parse time)."""
    return "flags" in inst.args_out


# --- Table 3-3: "ALU, basic" (I, tput 6) vs "ALU, basic, flagset" (F, tput 3)


def _alu_basic(inst):
    """ADD/SUB/ADC/SBC/NEG/... without flag setting.

    .. note::

        ``add2`` ("add <Xd>, <Xa>, <Xb>, <imm>") also lands here.  If that
        pattern is meant to be the extended-register form, Table 3-3 puts it
        under "ALU, extend and shift" (M, lat 2) instead.  Left in the basic
        row until an actual use is confirmed.
    """
    return isinstance(inst, AArch64BasicArithmetic) and not _sets_flags(inst)


def _alu_basic_flagset(inst):
    """ADDS/SUBS/ADCS/SBCS/NEGS/NGCS/..."""
    return isinstance(inst, AArch64BasicArithmetic) and _sets_flags(inst)


# --- Table 3-3 / 3-6: AArch64Logical mixes three different SWOG rows.


def _logical_basic(inst):
    """AND/BIC/EON/EOR/ORN/ORR/SBFX/UBFX/SXTB/UXTB/REV, no flags. (I, lat 1)"""
    return (
        isinstance(inst, AArch64Logical)
        and not _sets_flags(inst)
        and not isinstance(inst, (bfi, extr))
    )


def _logical_flagset(inst):
    """ANDS. (F, lat 1)"""
    return isinstance(inst, AArch64Logical) and _sets_flags(inst)


# --- Table 3-3: AArch64ConditionalSelect mixes CSEL, FCSEL and CMN.


def _cond_select(inst):
    """CSEL/CSINC/CSINV/CSNEG/CSET/CSETM/CNEG. (I, lat 1)"""
    return (
        isinstance(inst, AArch64ConditionalSelect)
        and not isinstance(inst, fcsel)
        and not _sets_flags(inst)
    )


def _cond_select_flagset(inst):
    """CMN, i.e. an ADDS alias. (F, lat 1)"""
    return isinstance(inst, AArch64ConditionalSelect) and _sets_flags(inst)


# --- Table 3-4: AArch64Multiply mixes MUL/MNEG (M) and MADD/MSUB (M0).


def _mul_plain(inst):
    """MUL, MNEG. (M, lat 2, tput 2)"""
    return isinstance(inst, (mul_xform, mneg_xform, mul_wform))


def _mul_accumulate(inst):
    """MADD, MSUB, UMADDL, UMSUBL. (M0, lat 2(1), tput 1)"""
    return isinstance(inst, (madd_xform, msub_xform, umaddl_wform))


# --- Table 3-3: AArch64ShiftedArithmetic mixes logical-shifted (I) and
#     arithmetic-shifted (I or M depending on shift type and amount).


def _logical_shifted(inst):
    """EOR/BIC with a shifted operand: "Logical, shift, no flagset". (I, lat 1)"""
    return isinstance(inst, AArch64ShiftedArithmetic) and inst.mnemonic.split(" ")[
        0
    ] in ("eor", "bic", "orr", "orn", "eon")


def _arith_shifted(inst):
    """ADD{S}/SUB{S} with a shifted operand.

    Table 3-3 splits this into "LSL shift <= 4" (I, lat 1) and
    "LSR/ASR/ROR shift or LSL shift > 4" (M, lat 2).  Distinguishing the two
    needs the barrel type and the immediate, which is not always a plain int
    in this model, so the conservative M/lat-2 row is used throughout.
    """
    return isinstance(inst, AArch64ShiftedArithmetic) and not _logical_shifted(inst)


# --- Table 3-15: the reduce rows differ by element count.


def _uaddlv_narrow(inst):
    """UADDLV with a 4H/4S source: "ASIMD arith, reduce, 4H/4S". (V13, lat 2)"""
    return isinstance(inst, vuaddlv_sform) and inst.datatype in ("4h", "4s")


def _uaddlv_wide(inst):
    """UADDLV with an 8B/8H/16B source. (V13[,V], lat 4)"""
    return isinstance(inst, vuaddlv_sform) and inst.datatype not in ("4h", "4s")


###############################################################################
#                                                                             #
# Opaque hooks                                                                #
#                                                                             #
###############################################################################


def add_further_constraints(slothy):
    """No extra microarchitectural constraints beyond the unit model.

    The Neoverse N1/V1 models restricted Neon instructions to issue slots
    0/1 because those cores only have two FP/ASIMD pipes.  The V2 has four
    (Figure 2-1), and the dispatch limits of section 4.1 (<=4 uOPs on S or B,
    <=4 on M, <=2 on M0, <=2 on V0, <=2 on V1, <=6 on L) are all at least as
    wide as the corresponding pipeline counts, so they are already implied by
    ``execution_units``.
    """
    _ = slothy


def has_min_max_objective(config):
    _ = config
    return False


def get_min_max_objective(slothy):
    _ = slothy


###############################################################################
#                                                                             #
# Execution units                                                             #
#                                                                             #
###############################################################################

execution_units = {
    # =====================================================================
    # Scalar load / store -- Tables 3-7, 3-8
    # =====================================================================
    (Ldr_X, Ldp_W, ldr_sxtw_wform): ExecutionUnit.L(),
    Ldp_X: ExecutionUnit.L01(),  # X-form pair: tput 2, not 3
    ldr_const: ExecutionUnit.L(),  # "Load register, literal": L, F
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): _GPR_STORE,
    # =====================================================================
    # Vector load / store -- Tables 3-13, 3-14, 3-19, 3-20
    # =====================================================================
    (
        Ldr_D,
        Ldr_Q,
        Ldp_Q,
        q_ld1_2,
        b_ldr_stack_with_inc,
        d_ldr_stack_with_inc,
        q_ldr1_stack,
        q_ldr1_post_inc,
    ): ExecutionUnit.L(),
    # Structure loads also occupy a V pipe (Table 3-19 lists "L, V"), but
    # modelling that as a cross product costs a lot of solver time for little
    # gain; the throughput is captured by inverse_throughput instead.
    (Ld2, Ld3, Ld4, q_ld2_lane_s, Q_Ld2_Lane_Post_Inc): ExecutionUnit.L(),
    (
        Str_Q,
        Stp_Q,
        St2,
        St3,
        St4,
        q_st1_4_with_postinc,
        d_str_stack_with_inc,
        d_stp_stack_with_inc,
        q_stp_stack_with_inc,
    ): _VEC_STORE,
    # =====================================================================
    # Scalar ALU -- Tables 3-3, 3-6
    # =====================================================================
    _alu_basic: ExecutionUnit.I(),
    _alu_basic_flagset: ExecutionUnit.F(),
    _logical_basic: ExecutionUnit.I(),
    _logical_flagset: ExecutionUnit.F(),
    _logical_shifted: ExecutionUnit.I(),
    _arith_shifted: ExecutionUnit.M(),
    _cond_select: ExecutionUnit.I(),
    _cond_select_flagset: ExecutionUnit.F(),
    AArch64ConditionalCompare: ExecutionUnit.F(),  # CCMN/CCMP
    AArch64LogicalShifted: ExecutionUnit.I(),  # "Logical, shift, no flagset"
    Tst: ExecutionUnit.F(),  # TST=ANDS, CMP=SUBS
    AArch64Shift: ExecutionUnit.I(),  # SBFM/UBFM, ASRV/LSLV/LSRV/RORV
    AArch64Move: ExecutionUnit.I(),  # MOVN/MOVK/MOVZ
    bfi: ExecutionUnit.M(),  # "Bitfield move, insert" BFM
    extr: ExecutionUnit.M(),  # "Bitfield extract, two regs": I, M
    (mov, movk_imm, movk_imm_lsl, lsr, lsr_wform, asr_wform, and_imm_wform, nop): (
        ExecutionUnit.I()
    ),
    # =====================================================================
    # Scalar multiply and CRC -- Tables 3-4, 3-22
    # =====================================================================
    _mul_plain: ExecutionUnit.M(),
    _mul_accumulate: ExecutionUnit.M0(),
    AArch64HighMultiply: ExecutionUnit.M(),  # SMULH/UMULH
    umull_wform: ExecutionUnit.M(),  # "Multiply long"
    AArch64CRC32: ExecutionUnit.M0(),
    # =====================================================================
    # FP transfer -- Table 3-12
    # =====================================================================
    (fmov_s_form, fmov_d_form): ExecutionUnit.V01(),  # vec -> gen reg
    (fmov_0, fmov_0_force_output): ExecutionUnit.M0(),  # gen -> low half
    (fmov_1, fmov_1_force_output): _GPR_TO_HIGH_HALF,  # gen -> high half
    fcsel: ExecutionUnit.V(),  # Table 3-11 "FP select"
    # =====================================================================
    # ASIMD arithmetic and logic -- Table 3-15
    # =====================================================================
    (vadd, vsub): ExecutionUnit.V(),  # "ASIMD arith, basic"
    uaddlp: ExecutionUnit.V(),  # "ASIMD arith, pair-wise"
    ASimdCompare: ExecutionUnit.V(),
    AArch64NeonLogical: ExecutionUnit.V(),
    _uaddlv_narrow: ExecutionUnit.V13(),
    _uaddlv_wide: ExecutionUnit.V13(),
    # =====================================================================
    # ASIMD miscellaneous -- Table 3-18
    # =====================================================================
    AArch64NeonCount: ExecutionUnit.V(),  # CLS/CLZ/CNT
    (Vzip, Vrev, Transpose): ExecutionUnit.V(),
    vext: ExecutionUnit.V(),
    vxtn: ExecutionUnit.V(),  # "ASIMD extract narrow"
    (vmov, vmovi): ExecutionUnit.V(),
    vbif: ExecutionUnit.V(),  # BIF/BIT/BSL
    # INS element to element (Table 3-18).  vmov_d is "mov <Dd>, <Va>.d[1]";
    # <Dd> is inferred as a NEON register, so this is a vector-to-vector move
    # ("ASIMD duplicate, element"), NOT a transfer to a general purpose register.
    (mov_b00, mov_d01, mov_vtov_d, vmov_d): ExecutionUnit.V(),
    (vtbl, vtbl_2): ExecutionUnit.V01(),  # TBL, 1 or 2 table regs
    VecToGprMov: ExecutionUnit.V01(),  # UMOV/SMOV, element -> gen reg
    (vdup, vdup_w): ExecutionUnit.M0(),  # "ASIMD duplicate, gen reg"
    (Vins, Mov_xtov_d, mov_wtov_s): _GPR_TO_ELEMENT,  # INS gen -> elem
    # =====================================================================
    # ASIMD multiply -- Table 3-15
    # =====================================================================
    (Vmul, Vqdmulh): ExecutionUnit.V02(),  # MUL/SQDMULH/SQRDMULH
    Vmla: ExecutionUnit.V02(),  # MLA/MLS
    Vmull: ExecutionUnit.V02(),  # SMULL/UMULL/SQDMULL
    Vmlal: ExecutionUnit.V02(),  # SMLAL/UMLAL/SMLSL/UMLSL
    # =====================================================================
    # ASIMD shift -- Table 3-15
    # =====================================================================
    VShiftImmediateBasic: ExecutionUnit.V13(),
    VShiftImmediateRounding: ExecutionUnit.V13(),  # SRSHR/URSHR, "complex"
    VShiftRegBasic: ExecutionUnit.V13(),  # SSHL/USHL
    AArch64NeonShiftInsert: ExecutionUnit.V13(),  # SLI/SRI
    vusra: ExecutionUnit.V13(),  # "ASIMD shift accumulate"
    # =====================================================================
    # Cryptography -- Table 3-21
    #
    # Covers AESE/AESD/AESMC/AESIMC and the 64x64 PMULL, all of which are
    # 2 cycles / tput 4 / V.  Note this is *not* the V23 row of Table 3-15,
    # which is the 8x8 polynomial multiply.
    # =====================================================================
    AESInstruction: ExecutionUnit.V(),
    (aesr_x2, aesr_x4, aese_x4): ExecutionUnit.V(),
    SHA3Instruction: ExecutionUnit.V0(),  # BCAX/EOR3/RAX1/XAR
    # =====================================================================
    # Virtual spill instructions
    # =====================================================================
    save: _GPR_STORE,  # GPR -> stack
    qsave: _VEC_STORE,  # NEON -> stack, behaves like a vector store
    (restore, qrestore): ExecutionUnit.L(),
}


###############################################################################
#                                                                             #
# Inverse throughput                                                          #
#                                                                             #
# 1 unless the SWOG throughput is lower than the number of pipelines in the    #
# corresponding symbol.                                                       #
#                                                                             #
###############################################################################

inverse_throughput = {
    # --- scalar load/store: L has 3 pipes / tput 3, L01 has 2 / tput 2 -----
    (Ldr_X, Ldp_X, Ldp_W, ldr_const, ldr_sxtw_wform): 1,
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): 1,
    # --- vector load: Table 3-13 / 3-19 -----------------------------------
    (Ldr_D, Ldr_Q): 1,
    Ldp_Q: 2,  # "Load vector pair, Q-form": tput 3/2
    q_ld1_2: 2,  # "ASIMD load, 1 element, 2 reg": tput 3/2
    (b_ldr_stack_with_inc, d_ldr_stack_with_inc): 1,
    (q_ldr1_stack, q_ldr1_post_inc): 1,  # LD1R: tput 3
    Ld2: 2,  # "LD2 multiple, Q-form": tput 3/2
    Ld3: 3,  # "LD3 multiple, Q-form": tput 1
    Ld4: 6,  # "LD4 multiple, Q-form": tput 1/2
    (q_ld2_lane_s, Q_Ld2_Lane_Post_Inc): 2,  # "LD2 one lane": tput 2
    # --- vector store: _VEC_STORE gives 2/cycle before scaling -------------
    Str_Q: 1,  # ST1 1 reg: tput 2
    Stp_Q: 2,  # STP Q-form / ST1 2 reg: tput 1
    q_st1_4_with_postinc: 4,  # ST1 4 reg Q-form: tput 1/2
    St2: 4,  # ST2 multiple Q-form: tput 1/2
    St3: 6,  # ST3 multiple Q-form: tput 1/3
    St4: 12,  # ST4 multiple Q-form B/H/S: tput 1/6
    (d_str_stack_with_inc, d_stp_stack_with_inc): 1,  # STR / STP D-form: tput 2
    q_stp_stack_with_inc: 2,  # STP Q-form: tput 1
    # --- scalar ALU: I/F/M pipe counts already match the SWOG throughput ---
    _alu_basic: 1,
    _alu_basic_flagset: 1,
    _logical_basic: 1,
    _logical_flagset: 1,
    _logical_shifted: 1,
    _arith_shifted: 1,
    _cond_select: 1,
    _cond_select_flagset: 1,
    AArch64ConditionalCompare: 1,
    AArch64LogicalShifted: 1,
    Tst: 1,
    AArch64Shift: 1,
    AArch64Move: 1,
    bfi: 1,
    extr: 1,
    (mov, movk_imm, movk_imm_lsl, lsr, lsr_wform, asr_wform, and_imm_wform, nop): 1,
    # --- scalar multiply / CRC --------------------------------------------
    _mul_plain: 1,
    _mul_accumulate: 1,
    AArch64HighMultiply: 1,
    umull_wform: 1,
    AArch64CRC32: 1,
    # --- FP transfer -------------------------------------------------------
    (fmov_s_form, fmov_d_form): 2,  # V01 has 2 pipes, tput 1
    (fmov_0, fmov_0_force_output): 1,
    (fmov_1, fmov_1_force_output): 1,
    fcsel: 1,
    # --- ASIMD -------------------------------------------------------------
    (vadd, vsub): 1,
    uaddlp: 1,
    ASimdCompare: 1,
    AArch64NeonLogical: 1,
    _uaddlv_narrow: 1,
    _uaddlv_wide: 1,
    AArch64NeonCount: 1,
    (Vzip, Vrev, Transpose): 1,
    vext: 1,
    vxtn: 1,
    (vmov, vmovi): 1,
    vbif: 1,
    (mov_b00, mov_d01, mov_vtov_d, vmov_d): 1,
    (vtbl, vtbl_2): 1,  # TBL 1-2 tables: V01, tput 2
    VecToGprMov: 2,  # UMOV/SMOV: V01, tput 1
    (vdup, vdup_w): 1,
    (Vins, Mov_xtov_d, mov_wtov_s): 1,
    (Vmul, Vqdmulh): 1,
    Vmla: 1,
    Vmull: 1,
    Vmlal: 1,
    VShiftImmediateBasic: 1,
    VShiftImmediateRounding: 1,
    VShiftRegBasic: 1,
    AArch64NeonShiftInsert: 1,
    vusra: 1,
    # --- crypto ------------------------------------------------------------
    AESInstruction: 1,
    aesr_x2: 2,  # wrapper for 2 AES rounds
    (aesr_x4, aese_x4): 4,  # wrapper for 4 AES rounds
    SHA3Instruction: 1,
    # --- virtual -----------------------------------------------------------
    (save, qsave): 1,
    (restore, qrestore): 1,
}


###############################################################################
#                                                                             #
# Latencies                                                                   #
#                                                                             #
###############################################################################

default_latencies = {
    # --- scalar load / store: Tables 3-7, 3-8 ------------------------------
    (Ldr_X, Ldp_X, Ldp_W, ldr_sxtw_wform): 4,
    ldr_const: 5,  # "Load register, literal"
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): 1,
    # --- vector load / store: Tables 3-13, 3-14, 3-19, 3-20 ----------------
    # Note the preamble of Table 3-13: vector loads need one extra cycle to
    # forward into the FP/ASIMD pipes, hence 6 rather than 4.
    (Ldr_D, Ldr_Q, Ldp_Q, q_ld1_2): 6,
    (b_ldr_stack_with_inc, d_ldr_stack_with_inc): 6,
    (q_ldr1_stack, q_ldr1_post_inc): 8,  # LD1R
    Ld2: 8,
    Ld3: 8,
    Ld4: 9,
    (q_ld2_lane_s, Q_Ld2_Lane_Post_Inc): 8,
    Str_Q: 2,
    Stp_Q: 2,
    q_st1_4_with_postinc: 2,
    St2: 4,
    St3: 6,
    St4: 7,
    (d_str_stack_with_inc, d_stp_stack_with_inc, q_stp_stack_with_inc): 2,
    # --- scalar ALU: Tables 3-3, 3-6 ---------------------------------------
    _alu_basic: 1,
    _alu_basic_flagset: 1,
    _logical_basic: 1,
    _logical_flagset: 1,
    _logical_shifted: 1,
    _arith_shifted: 2,
    _cond_select: 1,
    _cond_select_flagset: 1,
    AArch64ConditionalCompare: 1,
    AArch64LogicalShifted: 1,
    Tst: 1,
    AArch64Shift: 1,
    AArch64Move: 1,
    bfi: 2,  # BFM
    extr: 3,  # "Bitfield extract, two regs"
    (mov, movk_imm, movk_imm_lsl, lsr, lsr_wform, asr_wform, and_imm_wform, nop): 1,
    # --- scalar multiply / CRC: Tables 3-4, 3-22 ---------------------------
    _mul_plain: 2,
    _mul_accumulate: 2,  # 1 with accumulator forwarding, see get_latency
    AArch64HighMultiply: 3,
    umull_wform: 2,
    AArch64CRC32: 2,  # 1 with forwarding, see get_latency
    # --- FP transfer: Tables 3-11, 3-12 ------------------------------------
    (fmov_s_form, fmov_d_form): 2,
    (fmov_0, fmov_0_force_output): 3,
    (fmov_1, fmov_1_force_output): 5,
    fcsel: 2,
    # --- ASIMD: Tables 3-15, 3-18 ------------------------------------------
    (vadd, vsub): 2,
    uaddlp: 2,
    ASimdCompare: 2,
    AArch64NeonLogical: 2,
    _uaddlv_narrow: 2,
    _uaddlv_wide: 4,
    AArch64NeonCount: 2,
    (Vzip, Vrev, Transpose): 2,
    vext: 2,
    vxtn: 2,
    (vmov, vmovi): 2,
    vbif: 2,
    (mov_b00, mov_d01, mov_vtov_d, vmov_d): 2,
    (vtbl, vtbl_2): 2,
    VecToGprMov: 2,
    (vdup, vdup_w): 3,
    (Vins, Mov_xtov_d, mov_wtov_s): 5,
    (Vmul, Vqdmulh): 4,
    Vmla: 4,  # 1 with accumulator forwarding
    Vmull: 3,
    Vmlal: 4,  # 1 with accumulator forwarding
    VShiftImmediateBasic: 2,
    VShiftImmediateRounding: 4,
    VShiftRegBasic: 2,
    AArch64NeonShiftInsert: 2,
    vusra: 4,  # 1 with accumulator forwarding
    # --- crypto: Table 3-21 ------------------------------------------------
    AESInstruction: 2,
    (aesr_x2, aesr_x4, aese_x4): 2,
    SHA3Instruction: 2,
    # --- virtual -----------------------------------------------------------
    save: 1,
    qsave: 2,  # vector store latency
    restore: 4,
    qrestore: 6,  # vector load: one extra cycle to reach the V pipes
}


###############################################################################
#                                                                             #
# SLOTHY interface                                                            #
#                                                                             #
###############################################################################


def get_latency(src, out_idx, dst):
    """Latency of ``src`` as seen by ``dst``.

    Implements the accumulator late-forwarding paths documented in the notes
    of Tables 3-4, 3-15 and 3-22.

    .. note::

        Section 4.7 ("Region based fast forwarding") adds one cycle when the
        producer and consumer are not in the same forwarding region.  That is
        not modelled here; the latencies below are therefore the optimistic
        same-region figures.  ASIMD integer multiply/multiply-accumulate is
        explicitly listed as belonging to *no* region.
    """
    _ = out_idx  # out_idx unused

    instclass_src = find_class(src)
    instclass_dst = find_class(dst)

    latency = lookup_multidict(default_latencies, src, instclass_src)

    # Table 3-15 note 1: MLA/MLS accumulator forwarding, "4(1)".
    if (
        instclass_src in all_subclass_leaves(Vmla)
        and instclass_dst in all_subclass_leaves(Vmla)
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-15 note 1: SMLAL/UMLAL/SMLSL/UMLSL accumulator forwarding.
    if (
        instclass_src in all_subclass_leaves(Vmlal)
        and instclass_dst in all_subclass_leaves(Vmlal)
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-15 note 2: SSRA/USRA/SRSRA/URSRA accumulator forwarding.
    if (
        instclass_src is vusra
        and instclass_dst is vusra
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-4 note 2: MADD/MSUB accumulator forwarding, "2(1)".
    # The accumulator is the first input operand of madd/msub.
    if (
        _mul_accumulate(src)
        and _mul_accumulate(dst)
        and len(src.args_out) > 0
        and len(dst.args_in) > 0
        and src.args_out[0] == dst.args_in[0]
    ):
        return 1

    # Table 3-22 note 1: CRC32 result forwarding saves one cycle.
    if (
        instclass_src in all_subclass_leaves(AArch64CRC32)
        and instclass_dst in all_subclass_leaves(AArch64CRC32)
        and src.args_out[0] == dst.args_in[0]
    ):
        return 1

    return latency


def get_units(src):
    instclass_src = find_class(src)
    units = lookup_multidict(execution_units, src, instclass_src)
    if isinstance(units, list):
        return units
    return [units]


def get_inverse_throughput(src):
    instclass_src = find_class(src)
    return lookup_multidict(inverse_throughput, src, instclass_src)
