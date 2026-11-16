//! Flash Next's hyper-connection products: the hc up and down projections with their mix and norm epilogues.
const t = @import("types.zig");
const quant = @import("core").quant;
const affine_launch = @import("../launches/affine.zig");
const affine_fold = @import("../launches/affine_fold.zig");
const products = @import("project.zig");
const Ops = @import("ops.zig").Ops;
const counting = @import("../launches/util.zig").counting;
const Error = t.Error;
const Tensor = t.Tensor;
const Projection = quant.Projection;
const int = t.int;

/// A prompt's hc up projection and tf_fn_hc_mix16 with `xs` into `mixed` in one fold launch; false where not taken.
pub fn projectHcMix(o: Ops, x: Tensor, proj: Projection, m: usize, xs: Tensor, mixed: Tensor, dims: usize) Error!bool {
    const h = switch (proj.handle) {
        .mlx => |h| h,
        .dense, .bf16 => return false,
    };
    const w = products.affineOf(proj, h);
    try w.check();
    const z = o.l;
    if (!o.prefill or m == 0 or x.kind != .bf16 or xs.kind != .bf16 or mixed.kind != .bf16 or !o.bf16) return false;
    if (!z.affine.matrix or !z.affine.matrix_fold or (w.bits != 4 and w.bits != 5) or w.partial) return false;
    if (w.n != 4 * dims or dims % 32 != 0 or w.k % 64 != 0) return false;
    const shape = products.shapeOf(o, x, w, m);
    const tile = z.affine.choose(.project, .prefill, shape) orelse return false;
    if (!products.matrixTile(tile.id) or !tile.roundsAct(shape)) return false;
    if (o.stream == counting) return true;
    const kind = w.tables.table();
    try affine_fold.foldMixGo(&z.affine, z.r, .{
        .x = x.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = kind },
        .bias = .{ .p = w.bias, .kind = kind },
        .out = 0,
        .m = int(m),
        .n = int(w.n),
        .k = int(w.k),
        .bits = w.bits,
        .group = w.group,
        .fp16 = 0,
        .out16 = 0,
    }, xs.ptr, mixed.ptr, @intCast(dims), o.stream);
    return true;
}

/// A prompt's hc down projection (fp32) of `h` normed by `rinv` and `scale` in the tile, x's bits; null if not taken.
pub fn projectHcDownNorm(o: Ops, h: Tensor, rinv: u64, scale: u64, proj: Projection, m: usize, streams: usize, dims: usize) Error!?Tensor {
    const hd = switch (proj.handle) {
        .mlx => |v| v,
        .dense, .bf16 => return null,
    };
    const w = products.affineOf(proj, hd);
    try w.check();
    const z = o.l;
    if (!o.prefill or m == 0 or h.kind != .bf16 or !o.bf16 or w.partial or w.k != streams * dims or dims % 64 != 0) return null;
    const shape = products.shapeOf(o, h, w, m);
    const tile = z.affine.choose(.project, .prefill, shape) orelse return null;
    if (!products.matrixTile(tile.id)) return null;
    const kind = w.tables.table();
    var arg: affine_launch.Arg = .{
        .x = h.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = kind },
        .bias = .{ .p = w.bias, .kind = kind },
        .out = 0,
        .m = int(m),
        .n = int(w.n),
        .k = int(w.k),
        .bits = w.bits,
        .group = w.group,
        .fp16 = 0,
        .out16 = 0,
    };
    const w16 = (try affine_fold.copyFor(&z.affine, z.r, arg, o.stream)) orelse return null;
    const out = try o.arena.take(m * w.n * 4);
    arg.out = out;
    try affine_fold.fold16NormGo(&z.affine, z.r, arg, w16, rinv, scale, int(streams), int(dims), o.stream);
    return .{ .ptr = out, .kind = .f32 };
}

/// projectHcMix with the streams `h` normed by `rinv` and `scale` in place of their normed copy `xs`: the same bits.
pub fn projectHcMixNorm(o: Ops, x: Tensor, proj: Projection, m: usize, h: Tensor, rinv: u64, scale: u64, mixed: Tensor, dims: usize) Error!bool {
    const hd = switch (proj.handle) {
        .mlx => |v| v,
        .dense, .bf16 => return false,
    };
    const w = products.affineOf(proj, hd);
    try w.check();
    const z = o.l;
    if (!o.prefill or m == 0 or x.kind != .bf16 or h.kind != .bf16 or mixed.kind != .bf16 or !o.bf16) return false;
    if (!z.affine.matrix or !z.affine.matrix_fold or (w.bits != 4 and w.bits != 5) or w.partial) return false;
    if (w.n != 4 * dims or dims % 32 != 0 or w.k % 64 != 0) return false;
    const shape = products.shapeOf(o, x, w, m);
    const tile = z.affine.choose(.project, .prefill, shape) orelse return false;
    if (!products.matrixTile(tile.id) or !tile.roundsAct(shape)) return false;
    if (o.stream == counting) return true;
    const kind = w.tables.table();
    try affine_fold.foldMixNormGo(&z.affine, z.r, .{
        .x = x.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = kind },
        .bias = .{ .p = w.bias, .kind = kind },
        .out = 0,
        .m = int(m),
        .n = int(w.n),
        .k = int(w.k),
        .bits = w.bits,
        .group = w.group,
        .fp16 = 0,
        .out16 = 0,
    }, h.ptr, rinv, scale, mixed.ptr, @intCast(dims), o.stream);
    return true;
}
