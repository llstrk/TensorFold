//! The fold tiles' launches (fold=on, fold=bf16): hc mix and pair epilogues, tall items, BF16 weight copies.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const memory = @import("../memory.zig");
const util = @import("util.zig");
const affine = @import("affine.zig");
const Kernels = affine.Kernels;
const Arg = affine.Arg;
const BlockKind = affine.BlockKind;
const Error = runtime.Error;

/// fold=bf16's BF16 weight copies, decoded as wmma_fold stages them (its bits); past `limit` one stays on wmma_fold.
pub const Copies = struct {
    gpa: std.mem.Allocator,
    lock: std.atomic.Mutex = .unlocked,
    map: std.AutoHashMapUnmanaged(u64, ?memory.DeviceBuffer) = .{},
    bytes: usize = 0,
    /// The most bytes the copies may take.
    limit: usize,

    pub const min_rows = 512;

    pub fn deinit(c: *Copies) void {
        var it = c.map.valueIterator();
        while (it.next()) |v| if (v.*) |*b| b.free();
        c.map.deinit(c.gpa);
    }

    /// The copy of `a`'s weights, made on `s` if not yet; null where it is not taken.
    fn of(c: *Copies, r: *const runtime.Runtime, k: *const Kernels, a: Arg, s: abi.Stream) Error!?u64 {
        if (a.m < min_rows or a.fp16 != 0 or (a.bits != 4 and a.bits != 5) or @rem(a.k, 64) != 0) return null;
        while (!c.lock.tryLock()) std.Thread.yield() catch {};
        defer c.lock.unlock();
        const got = c.map.getOrPut(c.gpa, a.words) catch return null;
        if (got.found_existing) return if (got.value_ptr.*) |b| b.base() else null;
        got.value_ptr.* = null;
        const len = @as(usize, @intCast(a.n)) * @as(usize, @intCast(a.k)) * 2;
        if (c.bytes + len > c.limit) return null;
        var buf = memory.DeviceBuffer.alloc(r, len) catch return null;
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(buf.base());
        const total: usize = @as(usize, @intCast(a.n)) * @as(usize, @intCast(a.k));
        Kernels.go(r, k.folds.fold_decode[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(@intCast(total), 256) }, .{ .x = 256 }, s, &args) catch |e| {
            buf.free();
            return e;
        };
        got.value_ptr.* = buf;
        c.bytes += len;
        return buf.base();
    }
};

/// The fold tiles' kernels with epilogues, the tall plan's and fold=bf16's, 4 and 5 bits (gfx11's matrix cores).
pub const Functions = struct {
    /// wmma_fold for a hyper-connection's up projection with its stream mix as the epilogue, 4 and 5 bits (fold=on).
    wmma_fold_mix: [2]Function,
    /// ...and for a routed gate and up with the activation as the epilogue.
    wmma_fold_pair: [2]Function,
    /// The tall plan's items past 128 rows on the 256-row fold tile (product, gate and up); the 128-row kernels' bits.
    wmma_fold_tall: [2]Function,
    wmma_fold_pair_tall: [2]Function,
    /// wmma_fold16 and wmma_fold_mix reading x as the hc norm's (h, rinv, scale), the same bits.
    wmma_fold16_norm: [2]Function,
    wmma_fold_mix_norm: [2]Function,
    /// wmma_fold on a dense weight's BF16 copy (policy fold=bf16), 4 and 5 bits, and the decode that makes the copy.
    wmma_fold16: [2]Function,
    fold_decode: [2]Function,

    pub fn load(m: Module) Error!Functions {
        var f: Functions = undefined;
        inline for (.{ 4, 5 }, 0..) |bits, i| {
            f.wmma_fold_mix[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm20affine_wmma_fold_mixILi{d}EEEvNS0_6AffineEPKtPti", .{bits}));
            f.wmma_fold_pair[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm21affine_wmma_fold_pairILi{d}EEEvNS0_6AffineEPtif", .{bits}));
            f.wmma_fold16_norm[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_wmma_fold16_normILi{d}EEEvNS0_6AffineEPKtPKfS6_ii", .{bits}));
            f.wmma_fold_mix_norm[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm25affine_wmma_fold_mix_normILi{d}EEEvNS0_6AffineEPKtPKfS6_Pti", .{bits}));
            f.wmma_fold_tall[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm21affine_wmma_fold_tallILi{d}EEEvNS0_6AffineE", .{bits}));
            f.wmma_fold_pair_tall[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm26affine_wmma_fold_pair_tallILi{d}EEEvNS0_6AffineEPtif", .{bits}));
            f.wmma_fold16[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_wmma_fold16ILi{d}EEEvNS0_6AffineEPKt", .{bits}));
            f.fold_decode[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_fold_decodeILi{d}EEEvNS0_6AffineEPt", .{bits}));
        }
        return f;
    }
};

/// fold=bf16's copy of a prompt product's weights where blockGo would read it; null where it stays on wmma_fold.
pub fn copyFor(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!?u64 {
    if (!k.matrix or !k.matrix_fold or a.route.items != 0) return null;
    const c = k.copies orelse return null;
    if (s == util.counting) return if (a.m >= Copies.min_rows and a.fp16 == 0 and (a.bits == 4 or a.bits == 5) and @rem(a.k, 64) == 0) 1 else null;
    return c.of(r, k, a, s);
}

/// wmma_fold16 over `w16` (copyFor's) with x read as the streams a.x normed by `rinv` and `scale` (HcNorm).
pub fn fold16NormGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, w16: u64, rinv: u64, scale: u64, streams: c_int, dims: c_int, s: abi.Stream) Error!void {
    if (!k.matrix or (a.bits != 4 and a.bits != 5) or a.k != streams * dims or @rem(dims, 64) != 0) return Kernels.refuse("fold16 norm");
    var args: hl.Args = .{};
    inline for (.{ a, w16, rinv, scale, streams, dims }) |v| try args.add(v);
    try Kernels.go(r, k.folds.wmma_fold16_norm[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(a.n, 128), .y = Kernels.cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
}

/// foldMixGo with the streams `h` normed by `rinv` and `scale` (HcNormed) in place of their normed copy.
pub fn foldMixNormGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, h: u64, rinv: u64, scale: u64, mixed: u64, dims: c_int, s: abi.Stream) Error!void {
    if (!k.matrix or !k.matrix_fold or (a.bits != 4 and a.bits != 5)) return Kernels.refuse("fold mix norm");
    var args: hl.Args = .{};
    inline for (.{ a, h, rinv, scale, mixed, dims }) |v| try args.add(v);
    try Kernels.go(r, k.folds.wmma_fold_mix_norm[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(a.n, 128), .y = Kernels.cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
}

/// The hc up projection's fold tile (n = 4 dims) with its stream mix: mixed (m, dims) from xs (m, n).
pub fn foldMixGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, xs: u64, mixed: u64, dims: c_int, s: abi.Stream) Error!void {
    if (!k.matrix or !k.matrix_fold or (a.bits != 4 and a.bits != 5)) return Kernels.refuse("fold mix");
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(xs);
    try args.add(mixed);
    try args.add(dims);
    try Kernels.go(r, k.folds.wmma_fold_mix[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(a.n, 128), .y = Kernels.cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
}

/// A routed gate and up and its activation into `out` (pairs, width) on the fold tile; false, nothing launched, if not.
pub fn pairFoldRun(k: *const Kernels, r: *const runtime.Runtime, a: Arg, limit: f32, items: c_int, pairs: c_int, out: u64, s: abi.Stream) Error!bool {
    if (!k.matrix or !k.matrix_fold or (a.bits != 4 and a.bits != 5) or a.fp16 != 0) return false;
    if (@mod(a.n, 128) != 0 or @mod(a.k, 64) != 0) return false;
    var shape = Kernels.shapeOf(a, items);
    shape.plan_pairs = @intCast(@max(pairs, 0));
    const e = k.choose(.routed, .prefill, shape) orelse return false;
    if (!std.mem.eql(u8, e.id, "mlx.routed.prefill.matrix")) return false;
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(out);
    try args.add(@divExact(a.n, 2));
    try args.add(limit);
    if (a.m > 128) {
        if (a.m > 256) return Kernels.refuse("tall plan");
        const grid: hl.Dim3 = .{ .x = Kernels.cdiv(a.n, 128), .y = 1, .z = @intCast(items) };
        try Kernels.go(r, k.folds.wmma_fold_pair_tall[@intCast(a.bits - 4)], grid, .{ .x = 512 }, s, &args);
        try Kernels.go(r, k.folds.wmma_fold_pair[@intCast(a.bits - 4)], grid, .{ .x = 256 }, s, &args);
        return true;
    }
    try Kernels.go(r, k.folds.wmma_fold_pair[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(a.n, 128), .y = Kernels.cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    return true;
}

/// blockGo's fold cases, true when launched: a product on its BF16 copy, or the tall plan (`b`: wmma_fold's index).
pub fn blockGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, kind: BlockKind, b: usize, args: *hl.Args) Error!bool {
    if (kind == .matrix and k.matrix_fold and items == 1 and a.route.items == 0 and s != util.counting) {
        if (k.copies) |c| if (try c.of(r, k, a, s)) |w16| {
            try args.add(w16);
            try Kernels.go(r, k.folds.wmma_fold16[@intCast(a.bits - 4)], .{ .x = Kernels.cdiv(a.n, 128), .y = Kernels.cdiv(a.m, 128) }, .{ .x = 256 }, s, args);
            return true;
        };
    }
    if (a.route.items == 0 or a.m <= 128) return false;
    if (kind != .matrix or !k.matrix_fold or (a.bits != 4 and a.bits != 5) or a.m > 256) return Kernels.refuse("tall plan");
    const grid: hl.Dim3 = .{ .x = Kernels.cdiv(a.n, 128), .y = 1, .z = @intCast(items) };
    try Kernels.go(r, k.folds.wmma_fold_tall[@intCast(a.bits - 4)], grid, .{ .x = 512 }, s, args);
    try Kernels.go(r, k.wmma_fold[b], grid, .{ .x = 256 }, s, args);
    return true;
}
