using ThermodynamicPauli
using Test

@testset "ThermodynamicPauli" begin

    # ════════════════════════════════════════════════════════════════
    @testset "Integer encoding" begin

        @testset "pauli_to_int / int_to_pauli round-trip" begin
            for s in ("I", "X", "Y", "Z", "XY", "ZZ", "XIY", "XYZXYZ", "ZIZ")
                @test int_to_pauli(pauli_to_int(s)) == s
            end
        end

        @testset "stripping in pauli_to_int" begin
            @test pauli_to_int("IXI") == pauli_to_int("X")
            @test pauli_to_int("IIXIYI") == pauli_to_int("XIY")
            @test pauli_to_int("III") == PauliInt(0)
        end

        @testset "support_length" begin
            using ThermodynamicPauli: support_length
            @test support_length(PauliInt(0)) == 0
            @test support_length(pauli_to_int("X")) == 1
            @test support_length(pauli_to_int("XY")) == 2
            @test support_length(pauli_to_int("XIY")) == 3
            @test support_length(pauli_to_int("ZIIIZ")) == 5
        end

        @testset "pauli_weight" begin
            @test pauli_weight(PauliInt(0)) == 0
            @test pauli_weight(pauli_to_int("X")) == 1
            @test pauli_weight(pauli_to_int("XY")) == 2
            @test pauli_weight(pauli_to_int("XIY")) == 2
            @test pauli_weight(pauli_to_int("ZIIIZ")) == 2
        end

        @testset "strip_int" begin
            using ThermodynamicPauli: strip_int, _encode_raw
            @test strip_int(PauliInt(0)) == PauliInt(0)
            # "IX" raw → strip → "X"
            @test strip_int(_encode_raw("IX")) == _encode_raw("X")
            # "IIY" raw → strip → "Y"
            @test strip_int(_encode_raw("IIY")) == _encode_raw("Y")
            # "XIY" already stripped
            @test strip_int(_encode_raw("XIY")) == _encode_raw("XIY")
        end
    end

    # ════════════════════════════════════════════════════════════════
    @testset "Pauli algebra" begin

        @testset "pauli_product single-qubit" begin
            for p in ("X", "Y", "Z")
                phase, r = pauli_product(p, p)
                @test r == "I" && phase ≈ 1
            end
            @test pauli_product("X", "Y") == (im, "Z")
            @test pauli_product("Y", "Z") == (im, "X")
            @test pauli_product("Z", "X") == (im, "Y")
            @test pauli_product("Y", "X") == (-im, "Z")
            @test pauli_product("Z", "Y") == (-im, "X")
            @test pauli_product("X", "Z") == (-im, "Y")
        end

        @testset "commutes" begin
            @test commutes("X", "X") == true
            @test commutes("X", "Y") == false
            @test commutes("XY", "YX") == true
            @test commutes("ZZ", "XI") == false
            @test commutes("ZZ", "XX") == true
        end

        @testset "all_pauli_windows" begin
            @test length(all_pauli_windows(1)) == 4
            @test length(all_pauli_windows(2)) == 16
        end
    end

    # ════════════════════════════════════════════════════════════════
    @testset "PauliOp construction & string access" begin
        op = PauliOp("X" => 1.0, "Y" => 0.0, "Z" => 2.0)
        @test length(op) == 2
        @test !haskey(op, "Y")
        @test op["X"] ≈ 1.0
        @test op["Z"] ≈ 2.0

        op2 = PauliOp("X" => 0.0)
        @test isempty(op2)

        op3 = PauliOp{Float64}()
        @test isempty(op3)

        # Integer access
        @test op[pauli_to_int("X")] ≈ 1.0
        @test haskey(op, pauli_to_int("Z"))
    end

    # ════════════════════════════════════════════════════════════════
    @testset "PauliOp arithmetic" begin

        a = PauliOp("X" => 2.0, "ZZ" => 3.0)
        b = PauliOp("X" => 1.0, "Y" => 5.0)

        @testset "scalar multiplication" begin
            r = 2.0 * a
            @test r["X"] ≈ 4.0
            @test r["ZZ"] ≈ 6.0
            @test a * 3.0 == 3.0 * a
            @test isempty(0.0 * a)
        end

        @testset "division" begin
            r = a / 2.0
            @test r["X"] ≈ 1.0
            @test r["ZZ"] ≈ 1.5
        end

        @testset "addition" begin
            r = a + b
            @test r["X"] ≈ 3.0
            @test r["ZZ"] ≈ 3.0
            @test r["Y"] ≈ 5.0
        end

        @testset "subtraction" begin
            r = a - b
            @test r["X"] ≈ 1.0
            @test r["ZZ"] ≈ 3.0
            @test r["Y"] ≈ -5.0
        end

        @testset "negation" begin
            r = -a
            @test r["X"] ≈ -2.0
            @test r["ZZ"] ≈ -3.0
        end

        @testset "cancellation removes zeros" begin
            c = PauliOp("X" => 1.0, "Y" => 2.0)
            d = PauliOp("X" => -1.0, "Z" => 3.0)
            r = c + d
            @test !haskey(r, "X")
            @test r["Y"] ≈ 2.0
            @test r["Z"] ≈ 3.0
        end

        @testset "type promotion" begin
            real_op = PauliOp("X" => 1.0)
            complex_op = PauliOp("X" => 1.0 + 0.0im)
            r = real_op + complex_op
            @test valtype(r) == ComplexF64
        end
    end

    # ════════════════════════════════════════════════════════════════
    @testset "UpdateRule" begin
        rule = UpdateRule(Dict("XZ" => Dict("YI" => 0.5, "XZ" => 0.8)))
        @test rule.window_size == 2

        @test_throws ArgumentError UpdateRule(Dict(
            "XZ" => Dict("YI" => 1.0),
            "X"  => Dict("Y" => 1.0),
        ))

        @test_throws ArgumentError UpdateRule(Dict(
            "XZ" => Dict("Y" => 1.0),
        ))
    end

    # ════════════════════════════════════════════════════════════════
    @testset "apply_rule" begin

        @testset "simple replacement" begin
            rule = UpdateRule(Dict("XZ" => Dict("YI" => 0.5, "XZ" => 0.8)))
            op = PauliOp("XZ" => 1.0)
            r = apply_rule(op, rule)
            @test r["Y"] ≈ 0.5
            @test r["XZ"] ≈ 0.8
        end

        @testset "identity passes through" begin
            rule = UpdateRule(Dict("XZ" => Dict("YI" => 1.0)))
            op = PauliOp("I" => 3.0)
            r = apply_rule(op, rule)
            @test r["I"] ≈ 3.0
            @test length(r) == 1
        end

        @testset "multi-term operator" begin
            rule = UpdateRule(Dict(
                "XI" => Dict("YZ" => 1.0),
                "IX" => Dict("ZY" => 1.0),
            ))
            op = PauliOp("X" => 1.0)
            r = apply_rule(op, rule)
            @test r["ZY"] ≈ 1.0
            @test r["YZ"] ≈ 1.0
        end

        @testset "coefficients accumulate" begin
            rule = UpdateRule(Dict(
                "XI" => Dict("XI" => 0.3),
                "IX" => Dict("IX" => 0.7),
            ))
            op = PauliOp("X" => 2.0)
            r = apply_rule(op, rule)
            @test r["X"] ≈ 2.0
        end

        @testset "zero results are dropped" begin
            rule = UpdateRule(Dict(
                "XI" => Dict("XI" => 1.0),
                "IX" => Dict("IX" => -1.0),
            ))
            op = PauliOp("X" => 1.0)
            r = apply_rule(op, rule)
            @test isempty(r)
        end

        @testset "max_weight truncation" begin
            rule = UpdateRule(Dict(
                "XI" => Dict("YZ" => 1.0),   # weight 2
                "IX" => Dict("IX" => 1.0),   # weight 1 after strip
            ))
            op = PauliOp("X" => 1.0)
            r = apply_rule(op, rule; max_weight=1)
            @test haskey(r, "X")
            @test !haskey(r, "YZ")
        end
    end

    # ════════════════════════════════════════════════════════════════
    @testset "Liouvillian" begin

        @testset "H = Z: commutator" begin
            H = PauliOp("Z" => 1.0)
            L = Liouvillian(H)
            @test L.window_size == 1

            # [Z, X] = 2iY
            r = apply_rule(PauliOp("X" => 1.0), L)
            @test r["Y"] ≈ 2im

            # [Z, Y] = -2iX
            r = apply_rule(PauliOp("Y" => 1.0), L)
            @test r["X"] ≈ -2im

            r = apply_rule(PauliOp("Z" => 1.0), L)
            @test isempty(r)
        end

        @testset "H = X" begin
            H = PauliOp("X" => 1.0)
            L = Liouvillian(H)

            # [X, Y] = 2iZ
            r = apply_rule(PauliOp("Y" => 1.0), L)
            @test r["Z"] ≈ 2im

            # [X, Z] = -2iY
            r = apply_rule(PauliOp("Z" => 1.0), L)
            @test r["Y"] ≈ -2im
        end

        @testset "H = ZZ: Ising interaction" begin
            H = PauliOp("ZZ" => 1.0)
            L = Liouvillian(H)
            @test L.window_size == 2

            # [ZZ, X]: two windows contribute
            r = apply_rule(PauliOp("X" => 1.0), L)
            @test haskey(r, "YZ")
            @test haskey(r, "ZY")
            @test r["YZ"] ≈ 2im
            @test r["ZY"] ≈ 2im
        end

        @testset "[H, H] = 0" begin
            H = PauliOp("ZZ" => 1.0, "X" => 0.5)
            L = Liouvillian(H)
            r = apply_rule(H, L)
            clean!(r, 1e-10)
            @test isempty(r)
        end

        @testset "double commutator" begin
            H = PauliOp("Z" => 1.0)
            L = Liouvillian(H)
            op = PauliOp("X" => 1.0 + 0im)
            r1 = apply_rule(op, L)       # [Z,X] = 2iY
            r2 = apply_rule(r1, L)       # [Z,2iY] = 2i*(-2iX) = 4X
            @test r2["X"] ≈ 4.0 + 0im
            @test length(r2) == 1
        end

        @testset "TFIM" begin
            H = PauliOp("ZZ" => -1.0, "X" => -0.5)
            L = Liouvillian(H)
            @test L.window_size == 2

            # [H, Z]: only X term contributes: [-0.5X, Z] = -0.5*(-2iY) = iY
            r = apply_rule(PauliOp("Z" => 1.0), L)
            clean!(r, 1e-10)
            @test r["Y"] ≈ 1.0im
        end

        @testset "linearity" begin
            H1 = PauliOp("Z" => 1.0)
            H2 = PauliOp("X" => 1.0)
            H_sum = PauliOp("Z" => 2.0, "X" => 3.0)

            op = PauliOp("Y" => 1.0 + 0im)
            r_sum = apply_rule(op, Liouvillian(H_sum))
            r_manual = 2.0 * apply_rule(op, Liouvillian(H1)) + 3.0 * apply_rule(op, Liouvillian(H2))

            for k in union(keys(r_sum), keys(r_manual))
                @test get(r_sum.terms, k, 0.0im) ≈ get(r_manual.terms, k, 0.0im)
            end
        end

        @testset "identity in H contributes nothing" begin
            H = PauliOp("I" => 5.0, "Z" => 1.0)
            L = Liouvillian(H)
            r = apply_rule(PauliOp("X" => 1.0), L)
            @test r["Y"] ≈ 2im
            @test length(r) == 1
        end
    end

    # ════════════════════════════════════════════════════════════════
    @testset "propagate with max_weight" begin
        H = PauliOp("ZZ" => -1.0, "X" => -0.5)
        L = Liouvillian(H)
        op = PauliOp("Z" => 1.0 + 0im)

        # Without truncation — operator grows
        r_full = propagate(op, [L, L, L, L, L]; clean_tol=1e-10)
        max_w_full = maximum(pauli_weight(k) for k in keys(r_full))

        # With max_weight=3 — bounded
        r_trunc = propagate(op, [L, L, L, L, L]; clean_tol=1e-10, max_weight=3)
        max_w_trunc = maximum(pauli_weight(k) for k in keys(r_trunc))

        @test max_w_full > 3
        @test max_w_trunc <= 3
        @test length(r_trunc) < length(r_full)
    end

    # ════════════════════════════════════════════════════════════════
    @testset "propagate basic" begin
        H = PauliOp("Z" => 1.0)
        L = Liouvillian(H)
        op = PauliOp("X" => 1.0 + 0im)
        r = propagate(op, [L, L])
        @test r["X"] ≈ 4.0 + 0im
        @test length(r) == 1
    end

    # ════════════════════════════════════════════════════════════════
    @testset "clean!" begin
        op = PauliOp("X" => 1e-15 + 0im, "Y" => 1.0 + 0im, "Z" => 1e-6 + 0im)
        clean!(op, 1e-10)
        @test !haskey(op, "X")
        @test haskey(op, "Y")
        @test haskey(op, "Z")
        clean!(op, 1e-3)
        @test !haskey(op, "Z")
    end

    # ════════════════════════════════════════════════════════════════
    @testset "State overlaps" begin
        op = PauliOp("I" => 0.5, "Z" => 0.3, "X" => 0.1, "ZIZ" => 0.05, "XY" => 0.2)

        @test overlap_zero(op) ≈ 0.85       # I + Z + ZIZ
        @test overlap_plus(op) ≈ 0.6        # I + X
        @test overlap_maxmixed(op) ≈ 0.5
    end

    # ════════════════════════════════════════════════════════════════
    @testset "pauli_size and exp_size" begin
        op = PauliOp("I" => 1.0, "X" => 2.0, "ZIZ" => 3.0, "XY" => 0.5)

        s = pauli_size(op)
        @test !haskey(s, "I")
        @test s["X"] ≈ 2.0
        @test s["XY"] ≈ 1.0
        @test s["ZIZ"] ≈ 6.0

        e = exp_size(op, 0.0)
        @test e == op

        e2 = exp_size(op, 1.0)
        @test e2["I"] ≈ 1.0
        @test e2["X"] ≈ 2.0 * exp(1)
        @test e2["XY"] ≈ 0.5 * exp(2)
        @test e2["ZIZ"] ≈ 3.0 * exp(2)
    end

    # ════════════════════════════════════════════════════════════════
    @testset "hilbert_schmidt" begin
        a = PauliOp("X" => 1.0+2.0im, "ZZ" => 0.5+0im, "Y" => 1.0+0im)
        b = PauliOp("X" => 3.0+0im, "ZZ" => 0.0+4.0im, "XIY" => 1.0+0im)
        @test hilbert_schmidt(a, b) ≈ 3.0 - 4.0im

        c = PauliOp("X" => 3.0+4.0im, "Z" => 1.0+0im)
        @test hilbert_schmidt(c, c) ≈ 26.0 + 0im

        d = PauliOp("X" => 1.0+0im)
        e = PauliOp("Y" => 1.0+0im)
        @test hilbert_schmidt(d, e) ≈ 0.0 + 0im
    end

    # ════════════════════════════════════════════════════════════════
    @testset "theta_norm" begin
        op = PauliOp("X" => 3.0, "ZIZ" => 4.0)
        θ = 0.5
        expected = sqrt(9 * exp(-1) + 16 * exp(-2))
        @test theta_norm(op, θ) ≈ expected
        @test theta_norm(op, 0.0) ≈ 5.0
    end

    # ════════════════════════════════════════════════════════════════
    @testset "Operator growth" begin
        H = PauliOp("ZZ" => -1.0, "X" => -0.5)
        L = Liouvillian(H)
        op = PauliOp("Z" => 1.0 + 0im)

        r1 = apply_rule(op, L)
        clean!(r1, 1e-10)
        slen1 = maximum(ThermodynamicPauli.support_length(k) for k in keys(r1))
        @test slen1 <= 2

        r = op
        for _ in 1:5
            r = apply_rule(r, L)
            clean!(r, 1e-10)
        end
        max_support = maximum(ThermodynamicPauli.support_length(k) for k in keys(r))
        @test max_support > 1
    end

    # ════════════════════════════════════════════════════════════════
    @testset "Integer encoding edge cases" begin
        # Very long string
        long = "X" * "I"^30 * "Z"
        op = PauliOp(long => 1.0)
        @test op[long] ≈ 1.0
        @test int_to_pauli(pauli_to_int(long)) == long

        # Single-qubit strings
        for p in ("X", "Y", "Z")
            @test int_to_pauli(pauli_to_int(p)) == p
        end
    end

end  # top-level testset
