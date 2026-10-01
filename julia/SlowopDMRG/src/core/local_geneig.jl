#=
===============================================================================
LOCAL GENERALIZED-EIGENVALUE PRIMITIVE
===============================================================================

At each two-site DMRG block, solve the generalized eigenvalue problem

    (-M) · v = λ · (I + θ² · C_H²) · v

as an alternative to the ordinary eigenvalue problem for -M + θ²·C_H² that
`:eigsolve` uses.

Rationale: the RHS operator (I + θ² · C_H²) is positive definite with
eigenvalues ≥ 1 for any θ ≥ 0, so the effective generalized spectrum is
bounded in magnitude by ‖M‖ regardless of θ² — the C_H² term can no longer
dilate the local spectrum by an arbitrary factor of θ². Useful when θ² is
large and the ordinary operator −M + θ²C_H² becomes numerically stiff.

Uses `KrylovKit.geneigsolve` targeting the smallest real generalized
eigenvalue. Krylov-subspace ortho-projection is applied to both A_op and B_op
so trivial P-subspace eigenvalues sit at `subspace_coeffs.ortho_penalty` (large
positive) while the physical smallest eigenvalue lives in the (I−P) subspace.

Uses only the 4-legged factored C_H² envs (see dmrg_sweeper.jl:LH/RH).
===============================================================================
=#

"""
    apply_M_local(psi_vec, psiinds, LM, RM, M_p, M_p1) -> Vector

Single-layer M matvec on a local two-site block: `LM · v · M[p] · M[p+1] · RM`.
"""
function apply_M_local(psi_vec, psiinds, LM, RM, M_p, M_p1)
    # See apply_CH2_local: place the block tensor on the environments' device for
    # a single-device contraction, return the result on the host. No-ops on CPU.
    psiT = movedevice(ITensor(psi_vec, psiinds...))
    result = noprime(LM * psiT * M_p * M_p1 * RM)
    return vec(cpuarray(array(result, psiinds...)))
end

"""
    local_step_geneig(initvec, psiinds,
                       LM, RM, M_p, M_p1,
                       LH, RH, CH_p, CH_p1, Ct_p, Ct_p1,
                       thetasq, Q, subspace_coeffs,
                       dispon, dlog, matvec_counter;
                       eig_kwargs)

Solve the local generalized eigenvalue problem

    A_op(v) = λ · B_op(v)

with (`scale = subspace_coeffs[1]`; `ortho_penalty` is IGNORED for :geneig)

    A_op(v) = -scale · P̃ M P̃ v            → P-eigenvalue = 0
    B_op(v) = v + θ² · P̃ C_H² P̃ v          → P-eigenvalue = 1

`P = QQ†` projects onto the Krylov-state subspace to orthogonalize against,
`P̃ = I − P`. B_op is intentionally written as `v + θ²·P̃C_H²P̃v` rather than
`P̃(I+θ²C_H²)P̃v + Pv` — both are algebraically the same operator but the
former has one fewer projection call and less GPU fp noise.

Why no ortho_penalty is needed: on the P subspace `-M P̃v = 0`, so A_op v = 0
and B_op v = v ⇒ generalized eigenvalue = 0. On the P̃ subspace the physical
generalized eigenvalue is `λ_phys = -⟨M⟩ / (1 + θ²⟨C_H²⟩)`, strictly < 0
whenever ⟨M⟩ > 0 (any physical state ⊥ Krylov). `:SR` then targets λ_phys
automatically without any penalty; the P subspace parks at exactly 0 above
it. This avoids amplifying GPU ComplexF64 fp noise by huge coefficients,
which was tripping KrylovKit's absolute-tolerance Hermiticity check.

# Arguments
- `initvec::Vector`: initial guess (typically vec of psi[p]·psi[p+1])
- `psiinds`: block indices for reshaping vec ↔ ITensor
- `LM, RM, M_p, M_p1`: M-envs and per-site M MPO tensors
- `LH, RH`: 4-legged factored C_H² envs
- `CH_p, CH_p1`: bottom-layer (un-squared) C_H per-site MPO tensors
- `Ct_p, Ct_p1`: top-layer tensors, i.e. the adjoint of the bottom layer. Equal
  to `CH_p, CH_p1` on the Hamiltonian path (C_H is Hermitian); the Floquet path
  passes `adjoint_mpo(C_U)` tensors so B_op sees C_U†C_U rather than C_U²
- `thetasq::DTYPE`: θ² multiplier on the C_H² branch of B_op
- `Q`: orthonormal basis (matrix) for the Krylov-state subspace to
  orthogonalize against, or `nothing`
- `subspace_coeffs::Tuple{DTYPE,DTYPE}`: `(scale, _)`; only `scale` is used —
  the second element (ortho_penalty) is intentionally ignored for :geneig
  because the operator structure already zeroes P-subspace eigenvalues
- `eig_kwargs::Dict`: forwarded to `KrylovKit.geneigsolve`

# Returns
`(En::Vector, gs::Vector{<:Vector}, info)` matching the layout of
`KrylovKit.eigsolve`.
"""
function local_step_geneig(
    initvec, psiinds,
    LM, RM, M_p, M_p1,
    LH, RH,
    CH_p, CH_p1, Ct_p, Ct_p1,
    thetasq::DTYPE,
    Q, subspace_coeffs::Tuple{DTYPE,DTYPE},
    dispon::Int,
    dlog::DebugLogger,
    matvec_counter::Union{Ref{Int},Nothing};
    eig_kwargs::Dict = Dict(),
)
    scale = subspace_coeffs[1]   # ortho_penalty (subspace_coeffs[2]) intentionally unused

    AB_op = (v) -> begin
        if matvec_counter !== nothing
            matvec_counter[] += 1
        end

        # Project v onto the physical subspace P̃ once. Downstream matvec results
        # are also projected back to P̃; the P-component of the OUTPUT is left
        # unchanged from v_par → A: 0, B: identity (i.e., v itself).
        if Q === nothing
            v_perp = v
        else
            v_perp = v .- Q * (Q' * v)
        end

        Mv = apply_M_local(v_perp, psiinds, LM, RM, M_p, M_p1)
        Av_perp = (-scale) .* Mv
        CH2v_perp = apply_CH2_local(v_perp, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
        Bv_perp = v_perp .+ thetasq .* CH2v_perp

        if Q !== nothing
            Av_perp .-= Q * (Q' * Av_perp)
            Bv_perp .-= Q * (Q' * Bv_perp)
        end

        # B(v) = v + θ²·P̃C_H²P̃v ; A(v) = -scale·P̃MP̃v (both zero on P)
        # Add the P-passthrough piece explicitly so the OUTPUT respects the
        # decomposition B_op(v_par) = v_par, A_op(v_par) = 0.
        v_par = Q === nothing ? v .* 0 : (v .- v_perp)
        Bv = Bv_perp .+ v_par   # v_perp + θ²·... + v_par = v + θ²·(P̃C_H²P̃)v
        Av = Av_perp            # 0 on P part; -scale·P̃MP̃ on P̃ part

        # Rescale BOTH operators by the same constant. A generalized eigenpair
        # solves A v = λ B v, so (cA) v = λ (cB) v has identical λ and v — this
        # is exact, not an approximation. What it buys: ‖B‖ grows like θ², and
        # KrylovKit's `checkhermitian`/`checkposdef` test the imaginary part of
        # ⟨v|B|v⟩ against an ABSOLUTE tolerance. Under :eps_constrained θ² is
        # bisected and can reach 1e9+ when the constraint is infeasible at the
        # current bond dimension; there ⟨v|B|v⟩ ≈ 1.9e9 carries ~6e-4 of
        # roundoff imaginary part — only 3e-13 relative, but enough to abort the
        # solve with "operator does not appear to be hermitian". Dividing by
        # (1 + θ²) keeps the diagonal O(1) so the absolute check stays valid.
        gscale = one(DTYPE) / (one(DTYPE) + thetasq)
        return (Av .* gscale, Bv .* gscale)
    end

    En, gs, info = geneigsolve(
        AB_op, initvec, 1, :SR;
        ishermitian = true,
        isposdef = true,
        verbosity = max(0, dispon - 1),
        eig_kwargs...,
    )

    # Renormalize to unit 2-norm. `geneigsolve` normalizes its eigenvectors in
    # the B-metric (⟨x|B|x⟩ = 1), so ‖x‖₂ depends on whatever scaling B carries —
    # it was ‖x‖₂ < 1 with B = I + θ²C†C, and rescaling B by 1/(1+θ²) above
    # changes it again by √(1+θ²). The eigenvalue is invariant either way, and
    # the sweep observables divide by ⟨gs|gs⟩, but `_run_dmrg_core` reports
    # `nu` and `tauinv` WITHOUT normalizing, so a B-metric vector silently
    # rescales those diagnostics. `local_step_qcqp`'s `_call_geneig_inner`
    # already normalizes; this makes the two paths agree.
    gs = [begin
              n = norm(g)
              n > 0 ? g ./ n : g
          end for g in gs]

    return En, gs, info
end
