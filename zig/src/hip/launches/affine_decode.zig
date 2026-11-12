//! The decode stream tiles' launches, with the stream tile's bits: one-round (tiles/stream_one.hpp), LDS (stream_lds).

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const affine = @import("affine.zig");
const stream_pick = @import("stream_pick.zig");

const Error = runtime.Error;

/// The one-round tile's waves a block and column groups a wave (kOneWaves, kOneGroups).
pub const one_waves = 4;
pub const one_groups = 4;
/// The LDS tile's waves a block (kLdsWaves).
pub const lds_waves = 4;

/// A one-round width's functions by the stream tile's rows and the lanes a column (16, 32).
const OneSet = [affine.stream_rows.len][2]Function;

/// The tiles' functions; a tile switched off leaves its products on the stream tile, the reference it must equal.
pub const Tiles = struct {
    one_on: bool = true,
    one: [affine.bit_widths.len]?OneSet = @splat(null),
    lds_on: bool = true,
    /// The LDS tile's 5-bit plain products and 4-bit gate|up pairs, BF16 only.
    lds_plain: ?Function = null,
    lds_pair: ?Function = null,
};

/// The functions of `m`, whose stream kernels take `dot`.
pub fn load(m: Module, comptime dot: []const u8) Error!Tiles {
    @setEvalBranchQuota(20_000);
    var t: Tiles = .{};
    inline for (affine.bit_widths, 0..) |bits, b| {
        if (bits == 4 or bits == 5) {
            var set: OneSet = undefined;
            inline for (affine.stream_rows, 0..) |rows, r| {
                inline for (.{ 4, 5 }, 0..) |log, l| {
                    set[r][l] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_oneINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, rows, log }));
                }
            }
            t.one[b] = set;
        }
    }
    if (comptime std.mem.eql(u8, dot, "7DotBF16")) {
        t.lds_plain = try m.function("_ZN2tf4rocm15affine_dot2_ldsINS0_7DotBF16ELi5ELi4ELi4ELi5EEEvNS0_6AffineENS0_11StreamSidesEi");
        t.lds_pair = try m.function("_ZN2tf4rocm22affine_dot2_lds_swigluINS0_7DotBF16ELi4ELi4ELi4ELi5EEEvNS0_6AffineENS0_11StreamSidesEi");
    }
    return t;
}

/// `a` on a decode tile where one applies; false leaves it to the stream tile.
pub fn go(k: *const affine.Kernels, r: *const runtime.Runtime, a: affine.Arg, sides: affine.StreamSides, lpc_log2: u5, ri: usize, total_n: c_int, items: c_int, s: abi.Stream) Error!bool {
    if (sides.count == 0 and sides.pair_cols == 0 and try oneGo(k, r, a, lpc_log2, ri, items, s)) return true;
    const f = ldsFunction(k, a, lpc_log2, total_n, items, sides.pair_cols > 0) orelse return false;
    try ldsGo(r, f, a, sides, items, s);
    return true;
}

/// Whether `streamGo` (plain, or a gate|up pair with `pair_cols`) runs `a` on the LDS tile.
pub fn ldsTakes(k: *const affine.Kernels, a: affine.Arg, pair_cols: c_int, total_n: c_int, items: c_int) bool {
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    return ldsFunction(k, a, lpc_log2, total_n, items, pair_cols > 0) != null;
}

/// The LDS tile where the stream pick is four rows at CB 4 and 32 lanes a column over 2 to 4 rounds; null elsewhere.
fn ldsFunction(k: *const affine.Kernels, a: affine.Arg, lpc_log2: u5, total_n: c_int, items: c_int, pair: bool) ?Function {
    if (!k.decode.lds_on or affine.stream_rows[stream_pick.rowTile(a.m)] != 4) return null;
    if (lpc_log2 != 5 or a.k > 4096 or (a.k >> 5) <= 32) return null;
    const tile = stream_pick.pick(k.caps.generation, a.m, lpc_log2, a.bits, total_n, items, pair);
    if (affine.stream_cbs[tile.at] != 4) return null;
    return if (pair and a.bits == 4) k.decode.lds_pair else if (!pair and a.bits == 5) k.decode.lds_plain else null;
}

/// The LDS tile's launch: plain products two column groups a wave, pairs one; a group's sides get their own blocks.
fn ldsGo(r: *const runtime.Runtime, f: Function, a: affine.Arg, sides_in: affine.StreamSides, items: c_int, s: abi.Stream) Error!void {
    const pair = sides_in.pair_cols > 0;
    var sides = sides_in;
    const per_group: c_int = if (pair) 2 else 4;
    const g: c_int = if (pair) 1 else 2;
    var blocks: c_int = ldsBlocks(if (pair) sides.pair_cols else a.n, per_group, g);
    if (sides.count > 0) {
        blocks = 0;
        for (0..@intCast(sides.count)) |i| {
            sides.first[i] = @intCast(blocks);
            blocks += ldsBlocks(sides.n[i], per_group, g);
        }
        sides.first[@intCast(sides.count)] = @intCast(blocks);
    }
    if (s == util.counting) return;
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(sides);
    try args.add(groupShift(a.group));
    const grid: hl.Dim3 = .{ .x = @intCast(blocks * cdiv(a.m, 4)), .y = 1, .z = @intCast(items) };
    const lds: u32 = @intCast(4 * (a.k >> 5) * 68);
    try hl.launch(f, .{ .grid = grid, .block = .{ .x = 32 * lds_waves }, .shared = lds }, .{ .r = r, .handle = s }, &args);
}

fn ldsBlocks(cols: c_int, per_group: c_int, groups: c_int) c_int {
    return cdiv(cdiv(cdiv(cols, per_group), groups), lds_waves);
}

/// The one-round tile (RDNA 3.5, 4 or 5 bits, a column's chunks within 16 or 32 lanes).
fn oneGo(k: *const affine.Kernels, r: *const runtime.Runtime, a: affine.Arg, lpc_log2: u5, ri: usize, items: c_int, s: abi.Stream) Error!bool {
    if (!k.decode.one_on or k.caps.generation != .rdna3_5 or !k.wmma) return false;
    const b = std.mem.indexOfScalar(c_int, &affine.bit_widths, a.bits) orelse return false;
    const set = k.decode.one[b] orelse return false;
    if (lpc_log2 < 4 or a.k >> 5 > @as(c_int, 1) << lpc_log2) return false;
    const rows = affine.stream_rows[ri];
    const cb: c_int = if (rows == 4) 4 else 2;
    const per_group = (@as(c_int, 32) >> lpc_log2) * cb;
    const blocks = cdiv(cdiv(cdiv(a.n, per_group), one_groups), one_waves);
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(groupShift(a.group));
    const grid: hl.Dim3 = .{ .x = @intCast(blocks * cdiv(a.m, rows)), .y = 1, .z = @intCast(items) };
    try affine.Kernels.go(r, set[ri][@as(usize, lpc_log2) - 4], grid, .{ .x = 32 * one_waves }, s, &args);
    return true;
}

fn groupShift(group: c_int) c_int {
    return switch (group) {
        32 => 0,
        64 => 1,
        else => 2,
    };
}

fn cdiv(n: c_int, by: c_int) c_int {
    return @divTrunc(n + by - 1, by);
}
