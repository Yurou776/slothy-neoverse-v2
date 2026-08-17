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
SLOTHY microarchitecture model for the Arm Neoverse V1 core.

All latency / throughput / pipeline data is taken from:

    Arm(R) Neoverse(TM) V1 Core Software Optimization Guide,
    Issue 6.0 (r1p2), PJDOC-466751330-9685.

Section / table numbers in the comments below refer to that document.

Scope
-----
Covers the AArch64 A64 NEON + scalar instruction classes exposed by
``aarch64_neon.py``, *plus* the minimal SVE instruction classes that model
currently defines (``sve_*`` / ``SVELd4`` / ``SVESt4``).  The SVE coverage is
matched to the parser, not to the whole ISA: only the predicate/scalar,
integer, contiguous load/store and structured LD4/ST4 rows that have a
corresponding parser class are modelled (SWOG sections 3.24-3.29).

SVE modelling caveats
---------------------
* The V1 cracks 256-bit SVE ops across paired FP/ASIMD pipes (section 4.17
  lists the sustainable ASIMD+SVE issue combinations, e.g. "1 SVE uop on V0
  and 2 ASIMD uops on V13").  This model uses the plain SWOG pipeline symbols
  (V0 / V01 / M0 / L01) for SVE ops and lets them share the ``VEC*`` enum with
  NEON, which captures V-pipe *pressure* against NEON but not the exact
  cracking rules.  Refine ``add_further_constraints`` if you need the precise
  mixing bandwidth.
* The predicate-equals-destination "+1 cycle" rule (Tables 3-41/3-42 note 1)
  and MOVPRFX fusion (section 4.19) are not modelled.

Conventions used when transcribing the SWOG tables
--------------------------------------------------
* "Utilized Pipelines" maps directly onto the ``ExecutionUnit`` classmethods,
  which mirror the legend in Table 3-1.
* SLOTHY wants *inverse* throughput (cycles a unit is held), while the SWOG
  gives throughput (instructions per cycle).  Most of the SWOG throughput
  figure is already expressed by the number of pipelines in the symbol
  (e.g. ``I`` has 4 pipes and tput 4), in which case the inverse throughput
  here is simply 1.  It is only >1 where the SWOG throughput is *lower* than
  the pipeline count, using ``inverse_throughput = round(pipes / tput)``.

Notable differences from the V2 model (kept as inline "V1 diff" comments)
-------------------------------------------------------------------------
* The V1 has only **two** Integer Single-Cycle pipes (Table 2-1 / Fig 2-1),
  so ``S`` = {INT0, INT1} and ``I`` = S + M = 4 pipes (V2 has 4 single-cycle
  pipes and ``I`` = 6).  There is no separate ``R``/``F`` narrowing: V1's
  Table 3-4 places *both* flag-setting and non-flag-setting basic ALU ops on
  ``I`` (tput 3 vs 4), so flag-setting scalar ops stay on ``I`` here.
* ``CCMN``/``CCMP``/``TST``/``CMN`` and the flag-setting ADDS/ANDS family are
  ``I`` on V1 (Table 3-4), not the ``F`` set the V2 model uses.
* FP transfer vec->gen reg (``FMOV``) is a single ``V1`` pipe on the V1
  (Table 3-19, tput 1), not ``V01``.
* ``FCSEL`` is ``V01`` on the V1 (Table 3-17), not ``V``.
* The 8x8 polynomial multiply (PMUL/PMULL) is ``V01`` on the V1 (Table 3-25),
  so the ``V23`` symbol the V2 model needed does not exist here.
* ``LDP`` X-form is tput 1 on the V1 (Table 3-13, ``L``), i.e. inverse tput 3.
* Structured stores ST2/ST3 are faster on the V1 than on the V2
  (Table 3-35: ST2 tput 1, ST3 tput 2/3 @ lat 5); ST4 is unchanged.

.. warning::

    Consider disabling the EOR3 fusion callback on this target.
    ``aarch64_neon.py`` installs ``veor.global_fusion_cb = eor3_fusion_cb()``,
    which merges two EORs into one EOR3.  On the V1, EOR3 is a Crypto SHA3 op
    (Table 3-37): pipeline ``V0``, throughput **1**, while a plain EOR is an
    ASIMD logical op on ``V`` with throughput 4.  Fusing therefore trades
    4 EOR/cycle for 1 EOR3/cycle -- a net loss for EOR-heavy code such as
    GF(2) arithmetic.
"""

from enum import Enum

from slothy.helper import lookup_multidict
from slothy.targets.aarch64.aarch64_neon import (

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
from slothy.targets.aarch64.aarch64_sve import (
    find_class,
    sve_ptrue,
    sve_whilelt,
    sve_movprfx,
    sve_dup_w,
    sve_dup_x,
    sve_add,
    sve_sub,
    sve_mul_pred,
    sve_smulh_pred,
    sve_addvl,
    add_imm,
    sve_incd,
    sve_inch,
    sve_ld1h,
    sve_ld1h_mulvl,
    sve_ld1rh,
    sve_st1h,
    sve_st1h_mulvl,
    SVELd4,
    SVESt4,
    SVEPermute,
)

# Section 4.1: the dispatch stage can process up to 8 MOPs per cycle.
issue_rate = 8
llvm_mca_target = "neoverse-v1"


class ExecutionUnit(Enum):
    """Execution units of the Neoverse V1 core.

    The 15 issue pipelines are listed in Figure 2-1 / Table 2-1.  The
    classmethods below reproduce the pipeline symbols of Table 3-1, one
    method per symbol.
    """

    BR0 = 0
    BR1 = 1

    INT0 = 2  # Integer Single-Cycle 0..1  (only TWO of these on the V1)
    INT1 = 3

    MUL0 = 4  # Integer Single/Multi-Cycle 0..1
    MUL1 = 5

    VEC0 = 6  # FP/ASIMD 0..3
    VEC1 = 7
    VEC2 = 8
    VEC3 = 9

    LS0 = 10  # Load/Store 0..1
    LS1 = 11
    LOAD2 = 12  # Load 2 (load only)

    STORE0 = 13  # Store data 0..1
    STORE1 = 14

    def __repr__(self):
        return self.name

    @classmethod
    def B(cls):  # noqa: E743
        """Branch 0/1"""
        return [cls.BR0, cls.BR1]

    @classmethod
    def S(cls):
        """Integer single cycle 0/1"""
        return [cls.INT0, cls.INT1]

    @classmethod
    def M(cls):
        """Integer single/multicycle 0/1"""
        return [cls.MUL0, cls.MUL1]

    @classmethod
    def M0(cls):
        """Integer multicycle 0"""
        return [cls.MUL0]

    @classmethod
    def I(cls):  # noqa: E743
        """Integer single cycle 0/1 and single/multicycle 0/1"""
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

# Table 3-15: GPR stores use an address uOP (L01) and a store-data uOP (D).
_GPR_STORE = [[ls, d] for ls in ExecutionUnit.L01() for d in ExecutionUnit.D()]

# Tables 3-23 / 3-35 / 3-46: vector (and SVE contiguous) stores take their data
# from the V01 pipes, not from the store-data pipes.
_VEC_STORE = [[ls, v] for ls in ExecutionUnit.L01() for v in ExecutionUnit.V01()]

# Table 3-31 "ASIMD transfer, gen reg to element": M0, V
_GPR_TO_ELEMENT = [[ExecutionUnit.MUL0, v] for v in ExecutionUnit.V()]

# Table 3-19 "FP transfer, from gen to high half of vec reg": M0, V
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


# --- Table 3-4: "ALU, basic" (I, tput 4) vs "ALU, basic, flagset" (I, tput 3).
#     On the V1 both rows are on the ``I`` pipe set, so the split below is kept
#     for parallelism with the V2 model but maps to the same unit.


def _alu_basic(inst):
    """ADD/SUB/ADC/SBC/NEG/... without flag setting. (I, tput 4)"""
    return isinstance(inst, AArch64BasicArithmetic) and not _sets_flags(inst)


def _alu_basic_flagset(inst):
    """ADDS/SUBS/ADCS/SBCS/... (I, tput 3)"""
    return isinstance(inst, AArch64BasicArithmetic) and _sets_flags(inst)


# --- Table 3-4: AArch64Logical mixes basic logical and ANDS/BICS.


def _logical_basic(inst):
    """AND/BIC/EON/EOR/ORN/ORR/SBFX/UBFX/SXTB/UXTB/REV, no flags. (I, lat 1)"""
    return (
        isinstance(inst, AArch64Logical)
        and not _sets_flags(inst)
        and not isinstance(inst, (bfi, extr))
    )


def _logical_flagset(inst):
    """ANDS/BICS (no shift): "ALU, basic, flagset" row -> I, lat 1."""
    return isinstance(inst, AArch64Logical) and _sets_flags(inst)


# --- Table 3-4: AArch64ConditionalSelect mixes CSEL, FCSEL and CMN.


def _cond_select(inst):
    """CSEL/CSINC/CSINV/CSNEG/CSET/CSETM/CNEG. (I, lat 1)"""
    return (
        isinstance(inst, AArch64ConditionalSelect)
        and not isinstance(inst, fcsel)
        and not _sets_flags(inst)
    )


def _cond_select_flagset(inst):
    """CMN, i.e. an ADDS alias -> "ALU, basic, flagset" -> I, lat 1."""
    return isinstance(inst, AArch64ConditionalSelect) and _sets_flags(inst)


# --- Table 3-7: AArch64Multiply mixes MUL/MNEG (M) and MADD/MSUB (M0).


def _mul_plain(inst):
    """MUL, MNEG. (M, lat 2, tput 2)"""
    return isinstance(inst, (mul_xform, mneg_xform, mul_wform))


def _mul_accumulate(inst):
    """MADD, MSUB, UMADDL, UMSUBL. (M0, lat 2(1), tput 1)"""
    return isinstance(inst, (madd_xform, msub_xform, umaddl_wform))


# --- Table 3-4: AArch64ShiftedArithmetic mixes logical-shifted (I) and
#     arithmetic-shifted (M).


def _logical_shifted(inst):
    """EOR/BIC/ORR/ORN/EON with a shifted operand: "Logical, shift, no flagset".
    (I, lat 1)"""
    return isinstance(inst, AArch64ShiftedArithmetic) and inst.mnemonic.split(" ")[
        0
    ] in ("eor", "bic", "orr", "orn", "eon")


def _arith_shifted(inst):
    """ADD{S}/SUB{S} with a shifted operand.

    Table 3-4 splits this into "LSL shift <= 4" (I, lat 1) and
    "extend and shift" / "LSR/ASR/ROR or LSL shift > 4" (M, lat 2).
    Distinguishing the two needs the barrel type and the immediate, which is
    not always a plain int in this model, so the conservative M/lat-2 row is
    used throughout.
    """
    return isinstance(inst, AArch64ShiftedArithmetic) and not _logical_shifted(inst)


# --- Table 3-25: the reduce rows differ by element count.


def _uaddlv_narrow(inst):
    """UADDLV with a 4H/4S source: "ASIMD arith, reduce, 4H/4S". (V13, lat 2)"""
    return isinstance(inst, vuaddlv_sform) and inst.datatype in ("4h", "4s")


def _uaddlv_wide(inst):
    """UADDLV with an 8B/8H/16B source. (V13[,V], lat 4)"""
    return isinstance(inst, vuaddlv_sform) and inst.datatype not in ("4h", "4s")


# --- Table 3-42: SVE MUL/SMULH latency and throughput depend on element size.


def _sve_elt(inst):
    """Element size letter ('b'/'h'/'s'/'d') of an SVE instruction."""
    dt = inst.datatype
    if isinstance(dt, list):
        dt = dt[0]
    return dt.lower() if dt else None


def _sve_mul_narrow(inst):
    """SVE MUL/SMULH/UMULH, B/H/S element size. (V0, lat 4, tput 1)"""
    return isinstance(inst, (sve_mul_pred, sve_smulh_pred)) and _sve_elt(inst) != "d"


def _sve_mul_wide(inst):
    """SVE MUL/SMULH/UMULH, D element size. (V0, lat 5, tput 1/2)"""
    return isinstance(inst, (sve_mul_pred, sve_smulh_pred)) and _sve_elt(inst) == "d"


###############################################################################
#                                                                             #
# Opaque hooks                                                                #
#                                                                             #
###############################################################################


def add_further_constraints(slothy):
    """No extra microarchitectural constraints beyond the unit model.

    Unlike the Neoverse N1 (two FP/ASIMD pipes, which forces Neon ops onto
    issue slots 0/1), the V1 has four FP/ASIMD pipes (Figure 2-1).  The
    dispatch limits of section 4.1 (<=4 uOPs on S or B, <=4 on M, <=2 on M0,
    <=2 on V0, <=2 on V1, <=6 on L) are all at least as wide as the
    corresponding pipeline counts, so they are already implied by
    ``execution_units``.

    The ASIMD+SVE issue-mixing bandwidth of section 4.17 (max 2 SVE uOPs/cycle,
    or 4 ASIMD, or 1 SVE + 2 ASIMD on the complementary pipe pair) is only
    approximated by sharing the ``VEC*`` enum between NEON and SVE; add an
    explicit restriction here if the exact cracking rules matter.
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
    # Scalar load / store -- Tables 3-13, 3-15
    # =====================================================================
    (Ldr_X, Ldp_W, ldr_sxtw_wform): ExecutionUnit.L(),
    Ldp_X: ExecutionUnit.L(),  # V1 diff: X-form pair is tput 1 on L (inv tput 3)
    ldr_const: ExecutionUnit.L(),  # "Load register, literal": L
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): _GPR_STORE,
    # =====================================================================
    # Vector load / store -- Tables 3-21, 3-23, 3-33, 3-35
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
    # Structure loads also occupy a V pipe (Table 3-33 lists "L, V"), but
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
    # Scalar ALU -- Table 3-4, 3-11
    # =====================================================================
    _alu_basic: ExecutionUnit.I(),
    add_imm: ExecutionUnit.I(),
    _alu_basic_flagset: ExecutionUnit.I(),  # V1 diff: flagset stays on I
    _logical_basic: ExecutionUnit.I(),
    _logical_flagset: ExecutionUnit.I(),  # V1 diff
    _logical_shifted: ExecutionUnit.I(),
    _arith_shifted: ExecutionUnit.M(),
    _cond_select: ExecutionUnit.I(),
    _cond_select_flagset: ExecutionUnit.I(),  # V1 diff
    AArch64ConditionalCompare: ExecutionUnit.I(),  # V1 diff: CCMN/CCMP are I
    AArch64LogicalShifted: ExecutionUnit.I(),  # "Logical, shift, no flagset"
    Tst: ExecutionUnit.I(),  # V1 diff: TST=ANDS, CMP=SUBS are I (tput 3)
    AArch64Shift: ExecutionUnit.I(),  # SBFM/UBFM, ASRV/LSLV/LSRV/RORV
    AArch64Move: ExecutionUnit.I(),  # MOVN/MOVK/MOVZ
    bfi: ExecutionUnit.M(),  # "Bitfield move, insert" BFM
    extr: ExecutionUnit.M(),  # "Bitfield extract, two regs": I, M
    (mov, movk_imm, movk_imm_lsl, lsr, lsr_wform, asr_wform, and_imm_wform, nop): (
        ExecutionUnit.I()
    ),
    # =====================================================================
    # Scalar multiply and CRC -- Tables 3-7, 3-39
    # =====================================================================
    _mul_plain: ExecutionUnit.M(),
    _mul_accumulate: ExecutionUnit.M0(),
    AArch64HighMultiply: ExecutionUnit.M(),  # SMULH/UMULH
    umull_wform: ExecutionUnit.M(),  # "Multiply long"
    AArch64CRC32: ExecutionUnit.M0(),
    # =====================================================================
    # FP transfer -- Table 3-19
    # =====================================================================
    (fmov_s_form, fmov_d_form): ExecutionUnit.V1(),  # V1 diff: vec->gen is V1
    (fmov_0, fmov_0_force_output): ExecutionUnit.M0(),  # gen -> low half
    (fmov_1, fmov_1_force_output): _GPR_TO_HIGH_HALF,  # gen -> high half
    fcsel: ExecutionUnit.V01(),  # V1 diff: Table 3-17 "FP select" is V01
    # =====================================================================
    # ASIMD arithmetic and logic -- Table 3-25
    # =====================================================================
    (vadd, vsub): ExecutionUnit.V(),  # "ASIMD arith, basic"
    uaddlp: ExecutionUnit.V(),  # "ASIMD arith, pair-wise"
    ASimdCompare: ExecutionUnit.V(),
    AArch64NeonLogical: ExecutionUnit.V(),
    _uaddlv_narrow: ExecutionUnit.V13(),
    _uaddlv_wide: ExecutionUnit.V13(),
    # =====================================================================
    # ASIMD miscellaneous -- Table 3-31
    # =====================================================================
    AArch64NeonCount: ExecutionUnit.V(),  # CLS/CLZ/CNT
    (Vzip, Vrev, Transpose): ExecutionUnit.V(),
    vext: ExecutionUnit.V(),
    vxtn: ExecutionUnit.V(),  # "ASIMD extract narrow"
    (vmov, vmovi): ExecutionUnit.V(),
    vbif: ExecutionUnit.V(),  # BIF/BIT/BSL
    # INS element to element (Table 3-31).  vmov_d is "mov <Dd>, <Va>.d[1]";
    # <Dd> is inferred as a NEON register, so this is a vector-to-vector move
    # ("ASIMD duplicate, element"), NOT a transfer to a general purpose register.
    (mov_b00, mov_d01, mov_vtov_d, vmov_d): ExecutionUnit.V(),
    (vtbl, vtbl_2): ExecutionUnit.V01(),  # TBL, 1 or 2 table regs
    VecToGprMov: ExecutionUnit.V01(),  # UMOV/SMOV, element -> gen reg
    (vdup, vdup_w): ExecutionUnit.M0(),  # "ASIMD duplicate, gen reg"
    (Vins, Mov_xtov_d, mov_wtov_s): _GPR_TO_ELEMENT,  # INS gen -> elem
    # =====================================================================
    # ASIMD multiply -- Table 3-25
    # =====================================================================
    (Vmul, Vqdmulh): ExecutionUnit.V02(),  # MUL/SQDMULH/SQRDMULH
    Vmla: ExecutionUnit.V02(),  # MLA/MLS
    Vmull: ExecutionUnit.V02(),  # SMULL/UMULL/SQDMULL
    Vmlal: ExecutionUnit.V02(),  # SMLAL/UMLAL/SMLSL/UMLSL
    # =====================================================================
    # ASIMD shift -- Table 3-25
    # =====================================================================
    VShiftImmediateBasic: ExecutionUnit.V13(),
    VShiftImmediateRounding: ExecutionUnit.V13(),  # complex shifts
    VShiftRegBasic: ExecutionUnit.V13(),  # SSHL/USHL
    AArch64NeonShiftInsert: ExecutionUnit.V13(),  # SLI/SRI
    vusra: ExecutionUnit.V13(),  # "ASIMD shift accumulate"
    # =====================================================================
    # Cryptography -- Table 3-37
    #
    # Covers AESE/AESD/AESMC/AESIMC and the 64x64 PMULL, all of which are
    # 2 cycles / tput 4 / V.  (The 8x8 poly multiply lives in Table 3-25 as a
    # V01 op and is handled by aarch64_neon's Vmul/Vmull mapping, not here.)
    # =====================================================================
    AESInstruction: ExecutionUnit.V(),
    (aesr_x2, aesr_x4, aese_x4): ExecutionUnit.V(),
    SHA3Instruction: ExecutionUnit.V0(),  # BCAX/EOR3/RAX1/XAR
    # =====================================================================
    # SVE predicate / scalar -- Tables 3-41, 3-42
    # =====================================================================
    sve_ptrue: ExecutionUnit.M0(),  # "Predicate set/initialize/find next"
    sve_whilelt: ExecutionUnit.M0(),  # "Loop control, based on GPR"
    (sve_addvl, sve_incd, sve_inch): ExecutionUnit.M0(),  # "Predicate counting scalar"
    (sve_dup_w, sve_dup_x): ExecutionUnit.M0(),  # "Duplicate, scalar form"
    sve_movprfx: ExecutionUnit.V01(),  # "Move prefix"
    (sve_add, sve_sub): ExecutionUnit.V01(),  # "Arithmetic, basic"
    SVEPermute: ExecutionUnit.V01(),
    _sve_mul_narrow: ExecutionUnit.V0(),  # MUL/SMULH, B/H/S element -> V0
    _sve_mul_wide: ExecutionUnit.V0(),  # MUL/SMULH, D element -> V0
    # =====================================================================
    # SVE load / store -- Tables 3-45, 3-46
    #
    # Contiguous loads sit on L01 (Table 3-45); the LD4 structure form also
    # lists V01 but is L-only here, matching the NEON structure-load
    # simplification above.  Both LD4 and ST4 are flagged "avoid" in
    # section 4.18 -- the very high inverse throughput reflects that.
    # =====================================================================
    (sve_ld1h, sve_ld1h_mulvl, sve_ld1rh): ExecutionUnit.L01(),
    SVELd4: ExecutionUnit.L01(),
    (sve_st1h, sve_st1h_mulvl): _VEC_STORE,
    SVESt4: _VEC_STORE,
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
    # --- scalar load/store: L has 3 pipes / tput 3 ------------------------
    (Ldr_X, Ldp_W, ldr_const, ldr_sxtw_wform): 1,
    Ldp_X: 3,  # V1 diff: "Load pair X-form": L, tput 1 -> round(3/1)
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): 1,
    # --- vector load: Table 3-21 / 3-33 -----------------------------------
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
    St2: 2,  # V1 diff: ST2 multiple Q-form: tput 1 -> round(2/1)
    St3: 3,  # V1 diff: ST3 multiple Q-form: tput 2/3 -> round(2/(2/3))
    St4: 12,  # ST4 multiple Q-form B/H/S: tput 1/6
    (d_str_stack_with_inc, d_stp_stack_with_inc): 1,  # STR / STP D-form: tput 2
    q_stp_stack_with_inc: 2,  # STP Q-form: tput 1
    # --- scalar ALU: I/M pipe counts already match the SWOG throughput -----
    _alu_basic: 1,
    add_imm: 1,
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
    (fmov_s_form, fmov_d_form): 1,  # V1 diff: single V1 pipe, tput 1
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
    VecToGprMov: 2,  # UMOV/SMOV: tput 1 (SWOG lists V; narrowed to V01)
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
    # --- SVE: Tables 3-41, 3-42, 3-45, 3-46 --------------------------------
    sve_ptrue: 1,  # M0, tput 1
    sve_whilelt: 2,  # M0, tput 1/2 -> round(1/(1/2))
    (sve_addvl, sve_incd, sve_inch): 1,  # M0, tput 1
    (sve_dup_w, sve_dup_x): 1,  # M0, tput 1
    sve_movprfx: 1,  # V01, tput 2
    (sve_add, sve_sub): 1,  # V01, tput 2
    SVEPermute: 1,
    _sve_mul_narrow: 1,  # V0, tput 1
    _sve_mul_wide: 2,  # V0, tput 1/2 -> round(1/(1/2))
    (sve_ld1h, sve_ld1h_mulvl, sve_ld1rh): 1,  # L01, tput 2
    SVELd4: 8,  # L01, tput 1/4 -> round(2/(1/4)); avoid (sec 4.18)
    (sve_st1h, sve_st1h_mulvl): 1,  # _VEC_STORE base 2, tput 2
    SVESt4: 18,  # _VEC_STORE base 2, tput 1/9; avoid (sec 4.18)
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
    # --- scalar load / store: Tables 3-13, 3-15 ---------------------------
    (Ldr_X, Ldp_X, Ldp_W, ldr_sxtw_wform): 4,
    ldr_const: 4,  # V1 diff: "Load register, literal" is lat 4 on the V1
    (Str_X, Stp_X, Stp_W, w_stp_with_imm_sp): 1,
    # --- vector load / store: Tables 3-21, 3-23, 3-33, 3-35 ---------------
    # Vector loads need one extra cycle to forward into the FP/ASIMD pipes,
    # hence 6 rather than 4 (Table 3-21 preamble).
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
    St3: 5,  # V1 diff: ST3 multiple Q-form latency is 5 on the V1
    St4: 7,
    (d_str_stack_with_inc, d_stp_stack_with_inc, q_stp_stack_with_inc): 2,
    # --- scalar ALU: Tables 3-4, 3-11 -------------------------------------
    _alu_basic: 1,
    add_imm: 1,
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
    # --- scalar multiply / CRC: Tables 3-7, 3-39 --------------------------
    _mul_plain: 2,
    _mul_accumulate: 2,  # 1 with accumulator forwarding, see get_latency
    AArch64HighMultiply: 3,
    umull_wform: 2,
    AArch64CRC32: 2,  # 1 with forwarding, see get_latency
    # --- FP transfer: Tables 3-17, 3-19 -----------------------------------
    (fmov_s_form, fmov_d_form): 2,
    (fmov_0, fmov_0_force_output): 3,
    (fmov_1, fmov_1_force_output): 5,
    fcsel: 2,
    # --- ASIMD: Tables 3-25, 3-31 -----------------------------------------
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
    # --- crypto: Table 3-37 -----------------------------------------------
    AESInstruction: 2,
    (aesr_x2, aesr_x4, aese_x4): 2,
    SHA3Instruction: 2,
    # --- SVE: Tables 3-41, 3-42, 3-45, 3-46 --------------------------------
    sve_ptrue: 2,
    sve_whilelt: 3,
    (sve_addvl, sve_incd, sve_inch): 2,
    (sve_dup_w, sve_dup_x): 3,
    sve_movprfx: 2,
    (sve_add, sve_sub): 2,
    SVEPermute: 2,
    _sve_mul_narrow: 4,  # B/H/S element
    _sve_mul_wide: 5,  # D element
    (sve_ld1h, sve_ld1h_mulvl, sve_ld1rh): 6,
    SVELd4: 12,
    (sve_st1h, sve_st1h_mulvl): 2,
    SVESt4: 11,
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
    of Tables 3-7, 3-25 and 3-39.

    .. note::

        Section 4.8 ("Region based fast forwarding") adds one cycle when the
        producer and consumer are not in the same forwarding region.  That is
        not modelled here; the latencies below are therefore the optimistic
        same-region figures.  ASIMD integer multiply/multiply-accumulate is
        explicitly listed as belonging to *no* region.

        The SVE "governing predicate equals destination -> +1 cycle" rule
        (Tables 3-41/3-42 note 1) is likewise not modelled.
    """
    _ = out_idx  # out_idx unused

    instclass_src = find_class(src)
    instclass_dst = find_class(dst)

    latency = lookup_multidict(default_latencies, src, instclass_src)

    # Table 3-25 note 1: MLA/MLS accumulator forwarding, "4(1)".
    if (
        instclass_src in all_subclass_leaves(Vmla)
        and instclass_dst in all_subclass_leaves(Vmla)
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-25 note 1: SMLAL/UMLAL/SMLSL/UMLSL accumulator forwarding.
    if (
        instclass_src in all_subclass_leaves(Vmlal)
        and instclass_dst in all_subclass_leaves(Vmlal)
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-25 note 2: SSRA/USRA/SRSRA/URSRA accumulator forwarding.
    if (
        instclass_src is vusra
        and instclass_dst is vusra
        and src.args_in_out[0] == dst.args_in_out[0]
    ):
        return 1

    # Table 3-7 note 2: MADD/MSUB accumulator forwarding, "2(1)".
    # The accumulator is the first input operand of madd/msub.
    if (
        _mul_accumulate(src)
        and _mul_accumulate(dst)
        and len(src.args_out) > 0
        and len(dst.args_in) > 0
        and src.args_out[0] == dst.args_in[0]
    ):
        return 1

    # Table 3-39 note 1: CRC32 result forwarding saves one cycle.
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