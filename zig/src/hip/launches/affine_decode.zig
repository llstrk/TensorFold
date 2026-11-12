//! The decode stream tiles' launches: the one-round tile (tiles/stream_one.hpp), with the stream tile's bits.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");

const Error = runtime.Error;

/// The one-round tile's waves a block and column groups a wave (kOneWaves, kOneGroups).
pub const one_waves = 4;
pub const one_groups = 4;

/// A one-round width's functions by the stream tile's rows and the lanes a column (16, 32).
const OneSet = [affine.stream_rows.len][2]Function;

/// The tiles' functions; a tile switched off leaves its products on the stream tile, the reference it must equal.
pub const Tiles = struct {
    one_on: bool = true,
    one: [affine.bit_widths.len]?OneSet = @splat(null),
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
    return t;
}

/// `a` on a decode tile where one applies; false leaves it to the stream tile.
pub fn go(k: *const affine.Kernels, r: *const runtime.Runtime, a: affine.Arg, sides: affine.StreamSides, lpc_log2: u5, ri: usize, items: c_int, s: abi.Stream) Error!bool {
    return sides.count == 0 and sides.pair_cols == 0 and try oneGo(k, r, a, lpc_log2, ri, items, s);
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
