//! The one-round tile (tiles/stream_one.hpp) writes the stream tile's bytes: plain and routed, 4 and 5 bits.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const Rig = @import("product.zig").Rig;
const Kernels = affine.Kernels;

/// K of 9 to 32 chunks (16 and 32 lanes a column), widths with a ragged last group, rows past one row block.
const ks = [_]c_int{ 288, 320, 512, 640, 1024 };
const ns = [_]c_int{ 2560, 1000, 37 };
const row_counts = [_]usize{ 1, 2, 3, 4, 7, 9 };
const most_rows = 9;

const Kind = enum { plain, routed };

fn compare(t: *Rig, rng: *data.Rng, kind: Kind, bits: c_int, group: c_int, k_dim: c_int, n: c_int, m: usize, poison: bool) !usize {
    var reference = t.kernels.*;
    reference.decode.one_on = false;
    const experts: usize = if (kind == .routed) 2 else 1;
    const nn: usize = @intCast(n);
    const kk: usize = @intCast(k_dim);
    const words_row = kk * @as(usize, @intCast(bits)) / 32;
    const groups = kk / @as(usize, @intCast(group));
    const hw = try data.fill(gpa, u32, experts * nn * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * nn * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * nn * groups, rng, data.makeBias);
    defer gpa.free(hb);
    const hx = try data.fill(gpa, u16, 2 * most_rows * kk, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    // an infinity in one row's x sends its columns through the tile's repair
    if (poison) hx[kk + 5] = if (t.fp16) 0x7C00 else 0x7F80;
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    const bytes = 2 * most_rows * nn * 4;
    var outs: [2]Buffer = .{ try Buffer.alloc(t.base, bytes), undefined };
    defer outs[0].free();
    outs[1] = try Buffer.alloc(t.base, bytes);
    defer outs[1].free();
    // routed: pair 2r of row r to expert 0, 2r + 1 to expert 1 for the first half of the rows (x_div 2)
    const rows: c_int = @intCast(m);
    const second: usize = (m + 1) / 2;
    const plan = [_]c_int{ 0, 0, rows, 1, rows, @intCast(second) };
    var members: [2 * most_rows]c_int = undefined;
    for (0..m) |i| members[i] = @intCast(2 * i);
    for (0..second) |i| members[m + i] = @intCast(2 * i + 1);
    var items = try Buffer.fromHost(t.base, @as([]const c_int, &plan));
    defer items.free();
    var mem = try Buffer.fromHost(t.base, @as([]const c_int, members[0 .. m + second]));
    defer mem.free();
    const r = &t.base.r;
    const s = t.base.stream.handle;
    for (outs, [_]*const Kernels{ t.kernels, &reference }) |out, kern| {
        try out.fill8(0xA5, null);
        var arg: affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = out.ptr,
            .m = rows,
            .n = n,
            .k = k_dim,
            .bits = bits,
            .group = group,
            .fp16 = @intFromBool(t.fp16),
        };
        if (kind == .routed) arg.route = .{ .items = items.ptr, .members = mem.ptr, .x_div = 2 };
        try kern.streamGo(r, arg, .{}, n, @intCast(experts), s);
        try t.base.stream.synchronize();
    }
    const d = try data.sameOnDevice(outs[0], outs[1], bytes);
    if (d != 0) std.debug.print("  one-round {t} {d}-bit g{d} k {d} n {d} rows {d}{s}: {d} words differ\n", .{ kind, bits, group, k_dim, n, m, if (poison) " (inf)" else "", d });
    return d;
}

test "the one-round tile writes the stream tile's bytes for plain and routed products" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    var rng: data.Rng = .{ .state = 0x51D2A7C3E9B04F61 };
    var diff: usize = 0;
    for ([_]Kind{ .plain, .routed }) |kind| {
        for ([_]c_int{ 4, 5 }) |bits| {
            for (ks) |k_dim| {
                const group: c_int = if (bits == 4) 32 else 64;
                for (ns) |n| {
                    for (row_counts) |m| diff += try compare(&rig, &rng, kind, bits, group, k_dim, n, m, false);
                }
            }
            diff += try compare(&rig, &rng, kind, bits, 64, 640, 2560, 4, true);
            diff += try compare(&rig, &rng, kind, bits, 128, 1024, 1000, 3, false);
        }
    }
    try base.expect(diff == 0, "one-round tile: {d} words differ from the stream tile", .{diff});
}
