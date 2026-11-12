//! The LDS tile (tiles/stream_lds.hpp) writes the stream tile's bytes: 5-bit plain products, 4-bit gate|up pairs.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const Rig = @import("product.zig").Rig;
const Kernels = affine.Kernels;

const most_rows = 8;

const Case = struct { pair: bool, k: c_int, n: c_int, m: usize, poison: bool = false };

fn compare(t: *Rig, rng: *data.Rng, c: Case) !usize {
    var reference = t.kernels.*;
    reference.decode.lds_on = false;
    const bits: c_int = if (c.pair) 4 else 5;
    const group: c_int = if (c.pair) 32 else 64;
    // routed pairs as a window routes them: ten pairs a row (x_div 10) over 24 experts, pair p on expert p % 24
    const experts: usize = if (c.pair) 24 else 1;
    const nn: usize = @intCast(c.n);
    const kk: usize = @intCast(c.k);
    const words_row = kk * @as(usize, @intCast(bits)) / 32;
    const groups = kk / @as(usize, @intCast(group));
    const hw = try data.fill(gpa, u32, experts * nn * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * nn * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * nn * groups, rng, data.makeBias);
    defer gpa.free(hb);
    const hx = try data.fill(gpa, u16, most_rows * kk, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    // an infinity in one row's x sends its columns through the tile's repair
    if (c.poison) hx[kk + 5] = if (t.fp16) 0x7C00 else 0x7F80;
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    const bytes = 10 * most_rows * nn * 4;
    var outs: [2]Buffer = .{ try Buffer.alloc(t.base, bytes), undefined };
    defer outs[0].free();
    outs[1] = try Buffer.alloc(t.base, bytes);
    defer outs[1].free();
    const rows: c_int = @intCast(c.m);
    const pairs = 10 * c.m;
    var plan: [3 * 24]c_int = undefined;
    var members: [10 * most_rows]c_int = undefined;
    var used: usize = 0;
    for (0..experts) |e| {
        const first = used;
        var p = e;
        while (p < pairs) : (p += experts) {
            members[used] = @intCast(p);
            used += 1;
        }
        plan[3 * e ..][0..3].* = .{ @intCast(e), @intCast(first), @intCast(used - first) };
    }
    var items = try Buffer.fromHost(t.base, @as([]const c_int, plan[0 .. 3 * experts]));
    defer items.free();
    var mem = try Buffer.fromHost(t.base, @as([]const c_int, members[0..@max(used, 1)]));
    defer mem.free();
    const r = &t.base.r;
    const s = t.base.stream.handle;
    const pair_cols: c_int = if (c.pair) @divExact(c.n, 2) else 0;
    const total_n: c_int = if (c.pair) pair_cols else c.n;
    for (outs, [_]*const Kernels{ t.kernels, &reference }, 0..) |out, kern, which| {
        try out.fill8(0xA5, null);
        var arg: affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = if (c.pair) 0 else out.ptr,
            .out16 = if (c.pair) out.ptr else 0,
            .m = rows,
            .n = c.n,
            .k = c.k,
            .bits = bits,
            .group = group,
            .fp16 = @intFromBool(t.fp16),
        };
        if (c.pair) arg.route = .{ .items = items.ptr, .members = mem.ptr, .x_div = 10 };
        // the case must take the LDS tile, and the reference must not
        try base.expect(@import("../../launches/affine_decode.zig").ldsTakes(kern, arg, pair_cols, total_n, @intCast(experts)) == (which == 0), "lds tile pick: {any}", .{c});
        try kern.streamGo(r, arg, .{ .pair_cols = pair_cols }, total_n, @intCast(experts), s);
        try t.base.stream.synchronize();
    }
    const d = try data.sameOnDevice(outs[0], outs[1], bytes);
    if (d != 0) std.debug.print("  lds tile {any}: {d} words differ\n", .{ c, d });
    return d;
}

test "the LDS tile writes the stream tile's bytes for 5-bit plain products and routed 4-bit gate|up pairs" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (rig.fp16) return error.SkipZigTest;
    var rng: data.Rng = .{ .state = 0x7A3C51D2E9B04F61 };
    var diff: usize = 0;
    // 2 to 4 rounds with a ragged last one, widths with a ragged last group, one and two row blocks
    for ([_]c_int{ 1088, 2560, 3008, 4096 }) |k_dim| {
        for ([_]c_int{ 16384, 12354 }) |n| {
            for ([_]usize{ 3, 4, 7, 8 }) |m| diff += try compare(&rig, &rng, .{ .pair = false, .k = k_dim, .n = n, .m = m });
        }
        for ([_]usize{ 3, 4, 8 }) |m| diff += try compare(&rig, &rng, .{ .pair = true, .k = k_dim, .n = 1280, .m = m });
    }
    diff += try compare(&rig, &rng, .{ .pair = false, .k = 2560, .n = 16384, .m = 4, .poison = true });
    diff += try compare(&rig, &rng, .{ .pair = true, .k = 2560, .n = 1280, .m = 4, .poison = true });
    try base.expect(diff == 0, "lds tile: {d} words differ from the stream tile", .{diff});
}

fn compareGroup(t: *Rig, rng: *data.Rng, k_dim: c_int, m: usize, out_half: bool) !usize {
    var reference = t.kernels.*;
    reference.decode.lds_on = false;
    const widths = [_]c_int{ 2048, 2048, 6144, 6144 };
    const kk: usize = @intCast(k_dim);
    const words_row = kk * 5 / 32;
    const groups = kk / 64;
    var bufs: [3 * widths.len]Buffer = undefined;
    for (widths, 0..) |n, i| {
        const nn: usize = @intCast(n);
        const hw = try data.fill(gpa, u32, nn * words_row, rng, data.makeWord);
        defer gpa.free(hw);
        const hs = try data.fill(gpa, u16, nn * groups, rng, data.makeScale);
        defer gpa.free(hs);
        const hb = try data.fill(gpa, u16, nn * groups, rng, data.makeBias);
        defer gpa.free(hb);
        bufs[3 * i] = try Buffer.fromHost(t.base, hw);
        bufs[3 * i + 1] = try Buffer.fromHost(t.base, hs);
        bufs[3 * i + 2] = try Buffer.fromHost(t.base, hb);
    }
    defer for (&bufs) |*b| b.free();
    const hx = try data.fill(gpa, u16, most_rows * kk, rng, data.makeXB);
    defer gpa.free(hx);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    const per = most_rows * 6144 * 4;
    var outs: [2][widths.len]Buffer = undefined;
    for (&outs) |*o| for (o) |*b| {
        b.* = try Buffer.alloc(t.base, per);
    };
    defer for (&outs) |*o| for (o) |*b| b.free();
    const r = &t.base.r;
    const s = t.base.stream.handle;
    for (outs, [_]*const Kernels{ t.kernels, &reference }) |o, kern| {
        var sides: [widths.len]affine.Side = undefined;
        for (&sides, widths, 0..) |*side, n, i| {
            try o[i].fill8(0xA5, null);
            side.* = .{ .words = bufs[3 * i].ptr, .scale = bufs[3 * i + 1].ptr, .bias = bufs[3 * i + 2].ptr, .n = n, .out = o[i].ptr };
        }
        const arg: affine.Arg = .{
            .x = x.ptr,
            .words = 0,
            .scale = .{ .p = 0, .kind = 1 },
            .bias = .{ .p = 0, .kind = 1 },
            .out = 0,
            .m = @intCast(m),
            .n = 6144,
            .k = k_dim,
            .bits = 5,
            .group = 64,
            .fp16 = 0,
        };
        try kern.groupGo(r, arg, &sides, out_half, s);
        try t.base.stream.synchronize();
    }
    var d: usize = 0;
    for (outs[0], outs[1]) |a, b| d += try data.sameOnDevice(a, b, per);
    if (d != 0) std.debug.print("  lds tile group k {d} rows {d} half {}: {d} words differ\n", .{ k_dim, m, out_half, d });
    return d;
}

test "the LDS tile writes the stream tile's bytes for a group of four 5-bit products" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (rig.fp16) return error.SkipZigTest;
    var rng: data.Rng = .{ .state = 0x1C3A57D2E9B04F61 };
    var diff: usize = 0;
    for ([_]c_int{ 2560, 4096 }) |k_dim| {
        for ([_]usize{ 3, 4, 8 }) |m| {
            for ([_]bool{ true, false }) |half| diff += try compareGroup(&rig, &rng, k_dim, m, half);
        }
    }
    try base.expect(diff == 0, "lds tile group: {d} words differ from the stream tile", .{diff});
}
