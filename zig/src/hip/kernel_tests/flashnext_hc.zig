//! Flash Next's hyper-connection kernels (256-thread norm, 16-bit gates' mix, rinv norm) against what they replace.

const std = @import("std");
const rig = @import("rig.zig");
const util = @import("../launches/util.zig");
const ops = @import("../ops/ops.zig");
const fx = @import("flashnext.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

test "tf_fn_hc_norm_v4 writes tf_fn_hc_norm's bytes, with and without a branch, non-finite rows included" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.kind() != 2) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x4C4 };
    const rows = 5;
    const s = 4;
    const eps: f32 = 1e-6;
    // a full 1,024-wide pass, a partial last pass, and the widest the kernels take
    for ([_]usize{ 2560, 1000, 4096 }) |d| {
        const w = s * d;
        const hh = try fx.acts(t, &rng, rows * w, 2.0);
        defer gpa.free(hh);
        hh[1 * w + 7] = 0x7fc0; // NaN in row 1, +Inf and -Inf in row 3
        hh[3 * w + 11] = 0x7f80;
        hh[3 * w + d + 5] = 0xff80;
        const hb = try fx.acts(t, &rng, rows * d, 1.0);
        defer gpa.free(hb);
        const hi = try fx.acts(t, &rng, rows * s, 1.0);
        defer gpa.free(hi);
        const scale = try gpa.alloc(f32, w);
        defer gpa.free(scale);
        for (scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
        var h = try t.upload(hh);
        defer h.free();
        var branch = try t.upload(hb);
        defer branch.free();
        var inject = try t.upload(hi);
        defer inject.free();
        var sc = try t.upload(scale);
        defer sc.free();
        var outs: [4]DeviceBuffer = undefined;
        for (&outs) |*b| b.* = try t.alloc(rows * w * 2);
        defer for (&outs) |*b| b.free();
        for ([_]bool{ true, false }) |with_branch| {
            for ([_]@TypeOf(t.on.fnx.hc_norm){ t.on.fnx.hc_norm, t.on.fnx.hc_norm_v4 }, 0..) |f, i| {
                var a: util.Args = .{};
                const b: u64 = if (with_branch) at(branch) else 0;
                const inj: u64 = if (with_branch) at(inject) else 0;
                inline for (.{ at(h), b, inj, at(outs[2 * i]), at(outs[2 * i + 1]), at(sc), @as(c_int, 2), @as(c_int, s), @as(c_int, @intCast(d)), eps, @as(c_int, 0), @as(i64, 0) }) |v| try a.add(v);
                const threads: u32 = if (i == 0) 1024 else 256;
                try t.on.go(f, util.dim(s, rows, 1), util.dim(threads, 1, 1), 0, t.stream.handle, &a);
            }
            try t.stream.synchronize();
            for (0..2) |j| {
                const want = try rig.download(u16, outs[j]);
                defer gpa.free(want);
                const got = try rig.download(u16, outs[2 + j]);
                defer gpa.free(got);
                try std.testing.expectEqualSlices(u16, want, got);
            }
        }
    }
}

test "tf_fn_hc_mix16 on gates cast to the activation type writes tf_fn_hc_mix's bytes of the fp32 gates" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x4C16 };
    const rows = 7;
    const s = 4;
    const d = 2560;
    const n = rows * s * d;
    const hg = try gpa.alloc(f32, n);
    defer gpa.free(hg);
    for (hg) |*v| v.* = rng.unit() * 12.0;
    // past bf16's range, ties to even, and the non-finite and signed-zero gates
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -0.0, 0.0, 3.0e38, 1.00390625, 1.01171875 }, 0..) |v, j| hg[j * 997] = v;
    const hx = try fx.acts(t, &rng, n, 2.0);
    defer gpa.free(hx);
    var g = try t.upload(hg);
    defer g.free();
    var x = try t.upload(hx);
    defer x.free();
    var g16 = try t.alloc(n * 2);
    defer g16.free();
    var want = try t.alloc(rows * d * 2);
    defer want.free();
    var got = try t.alloc(rows * d * 2);
    defer got.free();
    const kind: ops.Kind = if (t.bf16) .bf16 else .f16;
    try o.cast(.{ .ptr = at(g), .kind = .f32 }, .{ .ptr = at(g16), .kind = kind }, n);
    const k = t.kind();
    const st = t.stream.handle;
    try t.on.fn_hc_mix(@ptrFromInt(at(g)), @ptrFromInt(at(x)), @ptrFromInt(at(want)), k, rows, s, d, st);
    try t.on.fn_hc_mix16(@ptrFromInt(at(g16)), @ptrFromInt(at(x)), @ptrFromInt(at(got)), k, rows, s, d, st);
    try t.stream.synchronize();
    const hw = try rig.download(u16, want);
    defer gpa.free(hw);
    const hgot = try rig.download(u16, got);
    defer gpa.free(hgot);
    try std.testing.expectEqualSlices(u16, hw, hgot);
}

/// ops/common.hpp's float_to_bf16: nearest even (no NaN here).
fn bf16Nearest(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

test "tf_fn_hc_norm_rinv writes tf_fn_hc_norm's h, and its rinv rebuilds x's bytes as bf16((h * rinv) * scale)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.kind() != 2) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x4C17 };
    const rows = 5;
    const s = 4;
    const d = 2560;
    const w = s * d;
    const eps: f32 = 1e-6;
    const hh = try fx.acts(t, &rng, rows * w, 2.0);
    defer gpa.free(hh);
    const hb = try fx.acts(t, &rng, rows * d, 1.0);
    defer gpa.free(hb);
    const hi = try fx.acts(t, &rng, rows * s, 1.0);
    defer gpa.free(hi);
    const scale = try gpa.alloc(f32, w);
    defer gpa.free(scale);
    for (scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var h = try t.upload(hh);
    defer h.free();
    var h2 = try t.upload(hh);
    defer h2.free();
    var branch = try t.upload(hb);
    defer branch.free();
    var inject = try t.upload(hi);
    defer inject.free();
    var sc = try t.upload(scale);
    defer sc.free();
    var h_out = try t.alloc(rows * w * 2);
    defer h_out.free();
    var x_out = try t.alloc(rows * w * 2);
    defer x_out.free();
    var rinv = try t.alloc(rows * s * 4);
    defer rinv.free();
    var a: util.Args = .{};
    inline for (.{ at(h), at(branch), at(inject), at(h_out), at(x_out), at(sc), @as(c_int, 2), @as(c_int, s), @as(c_int, d), eps, @as(c_int, 0), @as(i64, 0) }) |v| try a.add(v);
    try t.on.go(t.on.fnx.hc_norm, util.dim(s, rows, 1), util.dim(1024, 1, 1), 0, t.stream.handle, &a);
    const P = ?*anyopaque;
    const CP = ?*const anyopaque;
    try t.on.fn_hc_norm_rinv(@as(CP, @ptrFromInt(at(h2))), @as(CP, @ptrFromInt(at(branch))), @as(CP, @ptrFromInt(at(inject))), @as(P, @ptrFromInt(at(h2))), @ptrFromInt(at(rinv)), @ptrFromInt(at(sc)), rows, s, d, eps, t.stream.handle);
    try t.stream.synchronize();
    const want_h = try rig.download(u16, h_out);
    defer gpa.free(want_h);
    const want_x = try rig.download(u16, x_out);
    defer gpa.free(want_x);
    const got_h = try rig.download(u16, h2);
    defer gpa.free(got_h);
    const got_r = try rig.download(f32, rinv);
    defer gpa.free(got_r);
    try std.testing.expectEqualSlices(u16, want_h, got_h);
    for (0..rows) |r| for (0..s) |si| for (0..d) |j| {
        const i = (r * s + si) * d + j;
        try std.testing.expectEqual(want_x[i], bf16Nearest((fx.value(t, got_h[i]) * got_r[r * s + si]) * scale[si * d + j]));
    };
}
