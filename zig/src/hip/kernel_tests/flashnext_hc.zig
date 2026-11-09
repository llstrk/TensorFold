//! Flash Next's hyper-connection norm on 256 threads (tf_fn_hc_norm_v4) against the 1,024-thread kernel it replaces.

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
