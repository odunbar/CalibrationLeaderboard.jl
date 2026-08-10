# score_based_lm — score network architectures
#
# Two builders behind one calling convention:  net(Z) with Z :: nx x B (Float32,
# standardised coordinates) returns an nx x B prediction of the DSM noise eps.
#
#   build_score_mlp(nx)   — L63 (nx = 3).  No spatial structure to exploit.
#   build_score_unet(nx)  — L96 (nx = 40 or 100).  Periodic ring, so a 1-D U-Net
#                           with CIRCULAR padding.
#
# Both are deliberately small: the statistics window holds only 1001-5001
# samples, which cannot support a 300k-parameter network.

using Flux
using NNlib
using Random

########################################################################
###############  L63: plain MLP  ######################################
########################################################################

function build_score_mlp(nx::Int; hidden::Int = 128, rng::AbstractRNG)
    init = (dims...) -> Flux.glorot_uniform(rng, dims...)
    return Chain(
        Dense(nx     => hidden, swish; init),
        Dense(hidden => hidden, swish; init),
        Dense(hidden => hidden, swish; init),
        Dense(hidden => nx;            init),
    )
end

########################################################################
###############  L96: periodic 1-D U-Net  #############################
########################################################################

# Flux's Conv has no circular padding mode (only Int / tuple / SamePad(), all
# zero-fill), so pad explicitly with NNlib.pad_circular before an unpadded Conv.
circpad(p::Int) = x -> NNlib.pad_circular(x, (p, p); dims = 1)

function cconv(k::Int, ch::Pair{Int, Int}; rng::AbstractRNG)
    init = (dims...) -> Flux.glorot_uniform(rng, dims...)
    return Chain(circpad(k ÷ 2), Conv((k,), ch; pad = 0, init))
end

# Two convs + GroupNorm + swish at one resolution level.
function conv_block(k::Int, cin::Int, cout::Int; groups::Int, rng::AbstractRNG)
    g = gcd(groups, cout)
    return Chain(
        cconv(k, cin => cout; rng),
        GroupNorm(cout, g, swish),
        cconv(k, cout => cout; rng),
        GroupNorm(cout, g, swish),
    )
end

# Fixed (non-trainable) positional channels.  Per-site standardisation is correct
# for L96 -- the forcing is spatially varying for vec/flux -- but it breaks the
# translation equivariance that a plain CNN enforces.  Without these channels the
# conv net cannot represent a site-dependent score at all.
function positional_channels(nx::Int)
    i = collect(0:(nx - 1))
    return Float32.(hcat(
        sin.(2pi .* i ./ nx), cos.(2pi .* i ./ nx),
        sin.(4pi .* i ./ nx), cos.(4pi .* i ./ nx),
    ))                                    # nx x 4
end

struct CircUNet{E1, E2, BT, D2, D1, HD, PS}
    e1::E1
    e2::E2
    bott::BT
    d2::D2
    d1::D1
    head::HD
    pos::PS
end

Flux.@layer CircUNet trainable = (e1, e2, bott, d2, d1, head)

function (m::CircUNet)(Z::AbstractMatrix)
    nx, B = size(Z)
    x  = reshape(Z, nx, 1, B)
    ps = repeat(reshape(m.pos, nx, 4, 1), 1, 1, B)
    h0 = cat(x, ps; dims = 2)                       # nx x 5 x B

    h1 = m.e1(h0)                                   # nx     x C1 x B
    h2 = m.e2(NNlib.meanpool(h1, (2,)))             # nx/2   x C2 x B
    hb = m.bott(NNlib.meanpool(h2, (2,)))           # nx/4   x C3 x B

    u2 = m.d2(cat(NNlib.upsample_nearest(hb, (2,)), h2; dims = 2))
    u1 = m.d1(cat(NNlib.upsample_nearest(u2, (2,)), h1; dims = 2))
    return reshape(m.head(u1), nx, B)
end

function build_score_unet(nx::Int; base::Int = 16, kernel::Int = 5,
                          groups::Int = 8, rng::AbstractRNG)
    nx % 4 == 0 || error("build_score_unet: nx must be divisible by 4 (got $nx)")
    c1, c2, c3 = base, 2base, 4base
    init = (dims...) -> Flux.glorot_uniform(rng, dims...)
    return CircUNet(
        conv_block(kernel, 5,       c1; groups, rng),
        conv_block(kernel, c1,      c2; groups, rng),
        conv_block(kernel, c2,      c3; groups, rng),
        conv_block(kernel, c3 + c2, c2; groups, rng),
        conv_block(kernel, c2 + c1, c1; groups, rng),
        Conv((1,), c1 => 1; init),
        positional_channels(nx),
    )
end

########################################################################
###############  Dispatch  ############################################
########################################################################

# nx = 3 has no spatial structure for convolutions; nx >= 40 is a periodic ring.
#
# base = 16 gives ~66k parameters at both nx = 40 and nx = 100 (conv parameter
# count is set by the channel widths, not by nx).  base = 32 would be ~264k
# against 1001-5001 samples, which is well past what this data can support --
# raise it only if the held-out DSM loss says the net is underfitting.
function build_score_net(nx::Int; rng::AbstractRNG, hidden::Int = 128, base::Int = 16)
    return nx <= 8 ? build_score_mlp(nx; hidden, rng) : build_score_unet(nx; base, rng)
end

########################################################################
###############  Score-model lifecycle  ###############################
########################################################################
# Requires score_gfdt.jl to be included first (NeuralScore / GaussianScore).

# Sentinel for :kgmm before any data exists (make_score_model is called before
# the outer LM loop, so there is no window yet to cluster).  Mirrors the
# :gaussian arm's use of `nothing`, but needs its own type since `nothing` is
# already claimed by :gaussian's dispatch below.
struct PendingKGMM <: ScoreModel end

# Built once per LM trajectory and carried across outer iterations so the network
# is warm-started rather than relearned.  The :gaussian and :kgmm arms are both
# stateless (rebuilt from the current window each iteration -- k-means is cheap
# enough not to need warm-starting), so they return a sentinel instead.
function make_score_model(kind::Symbol, nx::Int, cfg, rng_net::AbstractRNG)
    kind === :gaussian && return nothing
    kind === :kgmm && return PendingKGMM()
    net = build_score_net(nx; rng = rng_net, hidden = cfg.hidden, base = cfg.base)
    return NeuralScore(net; sigma = cfg.sigma, nx = nx)
end

# Refit on the current attractor.  For :dsm, first call uses `epochs_init`/
# `lr_init`; every later call is a short warm refit, which is what makes the
# per-iteration cost tolerable and lets the net integrate information across the
# LM trajectory.
#
# The :gaussian and :kgmm arms are stateless, so they are simply rebuilt from the
# current window each iteration -- dispatch on the type, not on `=== nothing`,
# because after the first call the caller is holding a GaussianScore / KGMMScore
# rather than the sentinel.
const _NO_LOSS = (; train_loss = NaN, val_loss = NaN)

refresh_score!(::Nothing, X::AbstractMatrix, cfg, ::AbstractRNG) =
    (GaussianScore(X; ridge = 1e-2), _NO_LOSS)

refresh_score!(::GaussianScore, X::AbstractMatrix, cfg, ::AbstractRNG) =
    (GaussianScore(X; ridge = 1e-2), _NO_LOSS)

refresh_score!(::PendingKGMM, X::AbstractMatrix, cfg, rng_net::AbstractRNG) =
    (KGMMScore(X, rng_net; ridge = cfg.kgmm_ridge), _NO_LOSS)

refresh_score!(::KGMMScore, X::AbstractMatrix, cfg, rng_net::AbstractRNG) =
    (KGMMScore(X, rng_net; ridge = cfg.kgmm_ridge), _NO_LOSS)

function refresh_score!(model::NeuralScore, X::AbstractMatrix, cfg, rng_net::AbstractRNG)
    first = !model.trained
    stats = train_score!(
        model, X;
        epochs       = first ? cfg.epochs_init : cfg.epochs_warm,
        lr           = first ? cfg.lr_init : cfg.lr_warm,
        batch        = cfg.batch,
        rng          = rng_net,
        weight_decay = cfg.weight_decay,
        val_frac     = cfg.val_frac,
    )
    return model, stats
end
