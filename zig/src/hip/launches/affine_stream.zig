//! The stream tile's launches: its round and relu^2 epilogues, the one-round tile.
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

/// The stream family's functions for width `bits` (index `b`) from `m`: plain, pair, relu^2, one-round.
pub fn resolve(k: *Kernels, comptime dot: []const u8, comptime bits: c_int, comptime b: usize, m: Module) Error!void {
    @setEvalBranchQuota(300_000);
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
    return streamWith(k, r, a, sides_in, total_n, items, s, false);
}

/// `streamGo` with Nemotron's relu^2 as the epilogue: `a.out16` takes relu(round(sum))^2 in the activation type.
pub fn relu2Go(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
    if (a.out16 == 0 or a.out != 0) return Kernels.refuse("relu2 output");
    return streamWith(k, r, a, .{}, a.n, items, s, true);
}

fn streamWith(
    k: *const Kernels,
    r: *const runtime.Runtime,
    a: Arg,
    sides_in: StreamSides,
    total_n: c_int,
    items: c_int,
    s: abi.Stream,
    relu2: bool,
) Error!void {
    const b = Kernels.bitIndex(a.bits).?;
    var sides = sides_in;
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    const ri = stream_pick.rowTile(a.m);
    if (!relu2 and sides.count == 0 and sides.pair_cols == 0) {
        if (try oneGo(k, r, a, lpc_log2, ri, items, s)) return;
    }
    const tile = stream_pick.pick(k.caps.generation, a.m, lpc_log2, a.bits, total_n, items, sides.pair_cols > 0);
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
    const table = if (sides.pair_cols > 0) &k.pairs else if (relu2) &k.relu2 else &k.stream;
    const f = table[b][ri][tile.at].?;
    // a column block's row blocks run next to each other on x, so the later ones read its weights from cache
    const grid = stream_layout.grid(k.stream_interleave, blocks, Kernels.cdiv(a.m, stream_rows[ri]), items);
    try Kernels.go(r, f, grid, .{ .x = 32 * stream_waves }, s, &args);
}

/// The one-round tile where it applies (RDNA3.5, 4 or 5 bits, 16 or 32 lanes a column), else false: same bits.
fn oneGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, lpc_log2: u5, ri: usize, items: c_int, s: abi.Stream) Error!bool {
    if (!k.stream_one or k.caps.generation != .rdna3_5 or !k.wmma) return false;
    const o = std.mem.indexOfScalar(c_int, &one_bits, a.bits) orelse return false;
    const nch = a.k >> 5;
    if (lpc_log2 < 4 or nch > (@as(c_int, 1) << lpc_log2)) return false;
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
