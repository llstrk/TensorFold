//! MLX affine weights requantized to 4 bits at their own groups, for a draft head that only proposes.

const std = @import("std");
const mlx = @import("mlx.zig");
const convert = @import("convert.zig");

const Allocator = std.mem.Allocator;
const Host = mlx.Host;

/// `h` at 4 bits a group: scale (hi - lo) / 15 and bias lo in bf16, codes rounded against them; allocated from `a`.
pub fn requant4(a: Allocator, h: Host) (Allocator.Error || error{UnexpectedTensor})!Host {
    if (h.words.rank != 2 or h.scales.rank != 2 or h.biases.rank != 2) return error.UnexpectedTensor;
    if (!convert.isFloat(h.scales.dtype) or !convert.isFloat(h.biases.dtype) or h.bits < 2 or h.bits > 8) return error.UnexpectedTensor;
    const rows_n = h.words.shape[0];
    const groups = h.scales.shape[1];
    const g: usize = h.group;
    const k = groups * g;
    if (k % 8 != 0 or h.words.shape[1] * 32 != k * h.bits) return error.UnexpectedTensor;
    const words4 = try a.alloc(u32, rows_n * k / 8);
    @memset(words4, 0);
    const sc = try a.alloc(u16, rows_n * groups);
    const bi = try a.alloc(u16, rows_n * groups);
    const vals = try a.alloc(f32, g);
    defer a.free(vals);
    const words_row = h.words.shape[1];
    const mask: u64 = (@as(u64, 1) << @intCast(h.bits)) - 1;
    for (0..rows_n) |r| for (0..groups) |q| {
        const t = r * groups + q;
        const s = convert.load(h.scales.dtype, h.scales.bytes, t);
        const b = convert.load(h.biases.dtype, h.biases.bytes, t);
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (vals, 0..) |*v, i| {
            const bit = (q * g + i) * h.bits;
            const at = r * words_row + (bit >> 5);
            const w0: u64 = std.mem.readInt(u32, h.words.bytes[4 * at ..][0..4], .little);
            const w1: u64 = if ((bit & 31) + h.bits > 32) std.mem.readInt(u32, h.words.bytes[4 * (at + 1) ..][0..4], .little) else 0;
            const code: f32 = @floatFromInt(((w0 | w1 << 32) >> @intCast(bit & 31)) & mask);
            v.* = code * s + b;
            lo = @min(lo, v.*);
            hi = @max(hi, v.*);
        }
        const s16 = bf16Bits((hi - lo) / 15.0);
        const b16 = bf16Bits(lo);
        sc[t] = s16;
        bi[t] = b16;
        const sf: f32 = @bitCast(@as(u32, s16) << 16);
        const bf: f32 = @bitCast(@as(u32, b16) << 16);
        for (vals, 0..) |v, i| {
            const code: u32 = if (sf > 0) @intFromFloat(std.math.clamp(@round((v - bf) / sf), 0.0, 15.0)) else 0;
            const c = r * k + q * g + i;
            words4[c / 8] |= code << @intCast(4 * (c % 8));
        }
    };
    return mlx.hostOf(words4, sc, bi, rows_n, 4, h.group);
}

/// The bf16 bits of `f`, rounded to nearest even (finite inputs).
fn bf16Bits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

test "requant4 keeps an 8-bit projection's values within half a 4-bit step" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rng = prng.random();
    const rows_n = 3;
    const k = 128;
    var words: [rows_n * k / 4]u32 = undefined;
    for (&words) |*w| w.* = rng.int(u32);
    var scales: [rows_n * 2]u16 = undefined;
    var biases: [rows_n * 2]u16 = undefined;
    for (&scales, &biases) |*sv, *bv| {
        sv.* = bf16Bits(0.001 + 0.01 * rng.float(f32));
        bv.* = bf16Bits(-0.5 * rng.float(f32));
    }
    const h8 = mlx.hostOf(&words, &scales, &biases, rows_n, 8, 64);
    const h4 = try requant4(a, h8);
    defer a.free(@as([]u32, @alignCast(std.mem.bytesAsSlice(u32, @constCast(h4.words.bytes)))));
    defer a.free(@as([]u16, @alignCast(std.mem.bytesAsSlice(u16, @constCast(h4.scales.bytes)))));
    defer a.free(@as([]u16, @alignCast(std.mem.bytesAsSlice(u16, @constCast(h4.biases.bytes)))));
    try std.testing.expectEqual(@as(u8, 4), h4.bits);
    var v8: [rows_n * k]f64 = undefined;
    var v4: [rows_n * k]f64 = undefined;
    try mlx.reference(h8, &v8);
    try mlx.reference(h4, &v4);
    for (v8, v4, 0..) |x, y, i| {
        const step: f64 = @floatCast(@as(f32, @bitCast(@as(u32, scales4(h4, i / 64)) << 16)));
        // half a step, the bias's bf16 rounding and the top code's clamp
        try std.testing.expect(@abs(x - y) <= 0.5 * step + 0.02 * @abs(x) + 1e-3);
    }
}

fn scales4(h: Host, t: usize) u16 {
    return std.mem.readInt(u16, h.scales.bytes[2 * t ..][0..2], .little);
}
