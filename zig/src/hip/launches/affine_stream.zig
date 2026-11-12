//! The stream tile's launches: its round, relu^2 and hc-gates epilogues, the one-round and LDS tiles, the hc mix.
const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const stream_pick = @import("stream_pick.zig");
const stream_layout = @import("stream_layout.zig");
const Kernels = affine.Kernels;
const Arg = affine.Arg;
const StreamSides = affine.StreamSides;
const Error = runtime.Error;
const stream_cbs = affine.stream_cbs;
const stream_rows = affine.stream_rows;
const stream_waves = affine.stream_waves;
const one_bits = affine.one_bits;
const one_waves = affine.one_waves;
const one_groups = affine.one_groups;
const lds_waves = affine.lds_waves;

/// The stream family's functions for width `bits` (index `b`) from `m`: plain, pair, relu^2, one-round, LDS, hc.
pub fn resolve(k: *Kernels, comptime dot: []const u8, comptime bits: c_int, comptime b: usize, m: Module) Error!void {
    @setEvalBranchQuota(300_000);
    if (comptime std.mem.eql(u8, dot, "7DotBF16")) {
        if (bits == 5) k.lds_plain = try m.function("_ZN2tf4rocm15affine_dot2_ldsINS0_7DotBF16ELi5ELi4ELi4ELi5EEEvNS0_6AffineENS0_11StreamSidesEi");
        if (bits == 4) k.lds_pair = try m.function("_ZN2tf4rocm22affine_dot2_lds_swigluINS0_7DotBF16ELi4ELi4ELi4ELi5EEEvNS0_6AffineENS0_11StreamSidesEi");
    }
    // bf16 activations only, and hc up's 10 chunks (16 lanes a column)
    if (bits == 5 and comptime std.mem.eql(u8, dot, "7DotBF16")) {
        inline for (stream_rows, 0..) |rows, r| {
            k.one_mix[r] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm19affine_dot2_one_mixINS0_{s}ELi5ELi{d}ELi4EEEvNS0_6AffineEiPKtPti", .{ dot, rows }));
            inline for (stream_cbs, 0..) |cb, c| {
                k.gates[r][c] = if (cb * rows <= 16)
                    try m.function(std.fmt.comptimePrint("_ZN2tf4rocm24affine_dot2_stream_gatesINS0_{s}ELi5ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, rows, cb }))
                else
                    null;
            }
        }
    }
    inline for (one_bits, 0..) |ob, o| {
        if (ob == bits) {
            inline for (stream_rows, 0..) |rows, r| {
                inline for (.{ 4, 5 }, 0..) |log, l| {
                    k.one[o][r][l] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_oneINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, rows, log }));
                }
            }
        }
    }
    inline for (stream_rows, 0..) |rows, r| {
        inline for (stream_cbs, 0..) |cb, c| {
            k.stream[b][r][c] = if (cb * rows <= 16)
                try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb0EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
            else
                null;
            k.pairs[b][r][c] = if (cb * rows <= 16)
                try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
            else
                null;
            k.relu2[b][r][c] = if (cb * rows <= 16)
                try m.function(
                    std.fmt.comptimePrint("_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}" ++
                        "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }),
                )
            else
                null;
            k.relu2_pair[b][r][c] = if (cb * rows <= 16)
                try m.function(
                    std.fmt.comptimePrint("_ZN2tf4rocm29affine_dot2_stream_relu2_pairINS0_{s}" ++
                        "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineES3_iii", .{ dot, bits, rows, cb }),
                )
            else
                null;
        }
    }
}

/// The stream tile on `total_n` columns (a group's sum); `sides` counts them off block by block when it has any.
pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream) Error!void {
    return streamWith(k, r, a, sides_in, total_n, items, s, .round);
}

/// Whether `gatesGo` takes the hc down projection `a`.
pub fn gatesFit(k: *const Kernels, a: Arg, low: c_int) bool {
    if (a.bits != 5 or a.fp16 != 0 or a.out16 == 0 or (a.out == 0 and a.n != low) or a.route.items != 0) return false;
    const tile = stream_pick.pick(k.caps.generation, a.m, lanesLog(a.k), a.bits, a.n, 1, false);
    return k.gates[stream_pick.rowTile(a.m)][tile.at] != null;
}

/// The hc down with tf_fn_hc_gates as its epilogue: `a.out16` the silu gates (m, low), `a.out` the inject gates.
pub fn gatesGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, low: c_int, s: abi.Stream) Error!void {
    if (!gatesFit(k, a, low)) return Kernels.refuse("hc gates");
    var sides: StreamSides = .{};
    sides.pair_cols = low;
    return streamWith(k, r, a, sides, a.n, 1, s, .gates);
}

/// `streamGo` with Nemotron's relu^2 as the epilogue: `a.out16` takes relu(round(sum))^2 in the activation type.
pub fn relu2Go(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
    if (a.out16 == 0 or a.out != 0) return Kernels.refuse("relu2 output");
    return streamWith(k, r, a, .{}, a.n, items, s, .relu2);
}

fn streamWith(
    k: *const Kernels,
    r: *const runtime.Runtime,
    a: Arg,
    sides_in: StreamSides,
    total_n: c_int,
    items: c_int,
    s: abi.Stream,
    epi: enum { round, relu2, gates },
) Error!void {
    const b = Kernels.bitIndex(a.bits).?;
    var sides = sides_in;
    // the gates' `pair_cols` is their low rank for the epilogue, not a pair: the tile is picked as a plain product
    const pair = epi != .gates and sides.pair_cols > 0;
    const lpc_log2 = lanesLog(a.k);
    const ri = stream_pick.rowTile(a.m);
    if (epi == .round and sides.count == 0 and !pair) {
        if (try oneGo(k, r, a, lpc_log2, ri, items, s)) return;
    }
    const tile = stream_pick.pick(k.caps.generation, a.m, lpc_log2, a.bits, total_n, items, pair);
    if (epi == .round) {
        if (try ldsGo(k, r, a, sides, lpc_log2, ri, stream_cbs[tile.at], pair, items, s)) return;
    }
    const per_block = tile.per_block;
    var blocks: c_uint = Kernels.cdiv(total_n, per_block);
    if (sides.count > 0) {
        blocks = 0;
        for (0..@intCast(sides.count)) |i| {
            sides.first[i] = @intCast(blocks);
            blocks += Kernels.cdiv(sides.n[i], per_block);
        }
        sides.first[@intCast(sides.count)] = @intCast(blocks);
    }
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(sides);
    try args.add(@as(c_int, lpc_log2));
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    const table = if (pair) &k.pairs else if (epi == .relu2) &k.relu2 else &k.stream;
    const f = if (epi == .gates) k.gates[ri][tile.at].? else table[b][ri][tile.at].?;
    // a column block's row blocks run next to each other on x, so the later ones read its weights from cache
    const grid = stream_layout.grid(k.stream_interleave, blocks, Kernels.cdiv(a.m, stream_rows[ri]), items);
    try Kernels.go(r, f, grid, .{ .x = 32 * stream_waves }, s, &args);
}

/// The one-round tile where it applies (RDNA3.5, 4 or 5 bits, 16 or 32 lanes a column), else false: same bits.
fn oneGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, lpc_log2: u5, ri: usize, items: c_int, s: abi.Stream) Error!bool {
    if (!oneFits(k, a, lpc_log2)) return false;
    const o = std.mem.indexOfScalar(c_int, &one_bits, a.bits).?;
    const rows = stream_rows[ri];
    const cb: c_int = if (rows == 4) 4 else 2;
    const per_group = (@as(c_int, 32) >> lpc_log2) * cb;
    const ngroups: c_int = @intCast(Kernels.cdiv(a.n, per_group));
    const waves: c_int = @intCast(Kernels.cdiv(ngroups, one_groups));
    const blocks = Kernels.cdiv(waves, one_waves);
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    const f = k.one[o][ri][@as(usize, lpc_log2) - 4];
    try Kernels.go(r, f, .{ .x = blocks * Kernels.cdiv(a.m, rows), .y = 1, .z = @intCast(items) }, .{ .x = 32 * one_waves }, s, &args);
    return true;
}

/// The LDS tile for four rows at CB 4, 32 lanes, 2 to 4 rounds (K 1056 to 4096), or null where it does not apply.
fn ldsFunction(k: *const Kernels, a: Arg, lpc_log2: u5, ri: usize, cb: c_int, pair: bool) ?Function {
    if (!k.stream_lds or stream_rows[ri] != 4 or cb != 4 or lpc_log2 != 5 or a.k > 4096 or (a.k >> 5) <= 32) return null;
    return if (pair and a.bits == 4) k.lds_pair else if (!pair and a.bits == 5) k.lds_plain else null;
}

/// Whether `streamGo` (plain, or a gate|up pair with `pair_cols`) runs `a` on the LDS tile.
pub fn ldsTakes(k: *const Kernels, a: Arg, pair_cols: c_int, total_n: c_int, items: c_int) bool {
    const lpc_log2 = lanesLog(a.k);
    const tile = stream_pick.pick(k.caps.generation, a.m, lpc_log2, a.bits, total_n, items, pair_cols > 0);
    return ldsFunction(k, a, lpc_log2, stream_pick.rowTile(a.m), stream_cbs[tile.at], pair_cols > 0) != null;
}

/// `ldsFunction`'s launch: plain products two column groups a wave, pairs one. False where it does not apply.
fn ldsGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, sides_in: StreamSides, lpc_log2: u5, ri: usize, cb: c_int, pair: bool, items: c_int, s: abi.Stream) Error!bool {
    const f = ldsFunction(k, a, lpc_log2, ri, cb, pair) orelse return false;
    var sides = sides_in;
    const per_group: c_int = if (pair) 2 else 4;
    const g: c_int = if (pair) 1 else 2;
    const blocksOf = struct {
        fn of(cols: c_int, pg: c_int, gw: c_int) c_uint {
            return Kernels.cdiv(@intCast(Kernels.cdiv(@intCast(Kernels.cdiv(cols, pg)), gw)), lds_waves);
        }
    }.of;
    var blocks: c_uint = blocksOf(if (pair) sides.pair_cols else a.n, per_group, g);
    if (sides.count > 0) {
        blocks = 0;
        for (0..@intCast(sides.count)) |i| {
            sides.first[i] = @intCast(blocks);
            blocks += blocksOf(sides.n[i], per_group, g);
        }
        sides.first[@intCast(sides.count)] = @intCast(blocks);
    }
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(sides);
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    const lds: u32 = @intCast(4 * (a.k >> 5) * 68);
    try Kernels.goShared(r, f, .{ .x = blocks * Kernels.cdiv(a.m, 4), .y = 1, .z = @intCast(items) }, .{ .x = 32 * lds_waves }, lds, s, &args);
    return true;
}

/// Whether the one-round tile takes a product of `lpc_log2` lanes a column.
fn oneFits(k: *const Kernels, a: Arg, lpc_log2: u5) bool {
    if (!k.stream_one or k.caps.generation != .rdna3_5 or !k.wmma) return false;
    if (std.mem.indexOfScalar(c_int, &one_bits, a.bits) == null) return false;
    return lpc_log2 >= 4 and (a.k >> 5) <= (@as(c_int, 1) << lpc_log2);
}

/// The lanes a column of the stream tile: K's chunks rounded up to a power of two, 32 at most.
fn lanesLog(k_dim: c_int) u5 {
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (k_dim >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    return lpc_log2;
}

/// Whether `oneMixGo` takes the hc up projection `a` (n = 4 dims).
pub fn oneMixFits(k: *const Kernels, a: Arg, dims: c_int) bool {
    return a.bits == 5 and a.n == 4 * dims and a.route.items == 0 and a.fp16 == 0 and lanesLog(a.k) == 4 and
        k.one_mix[stream_pick.rowTile(a.m)] != null and oneFits(k, a, 4);
}

/// The hc up on the one-round tile with tf_fn_hc_mix as its epilogue: `mixed` (m, dims) bf16 from `xs` (m, 4 dims).
pub fn oneMixGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, xs: u64, mixed: u64, dims: c_int, s: abi.Stream) Error!void {
    if (!oneMixFits(k, a, dims)) return Kernels.refuse("hc mix");
    const lpc_log2 = lanesLog(a.k);
    const ri = stream_pick.rowTile(a.m);
    const rows = stream_rows[ri];
    const ngroups: c_int = @intCast(Kernels.cdiv(dims, @as(c_int, 32) >> lpc_log2));
    const waves: c_int = @intCast(Kernels.cdiv(ngroups, one_groups));
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    try args.add(xs);
    try args.add(mixed);
    try args.add(dims);
    const f = k.one_mix[ri].?;
    try Kernels.go(r, f, .{ .x = Kernels.cdiv(waves, one_waves) * Kernels.cdiv(a.m, rows), .y = 1, .z = 1 }, .{ .x = 32 * one_waves }, s, &args);
}
