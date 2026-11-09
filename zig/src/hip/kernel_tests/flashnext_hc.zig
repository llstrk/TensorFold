//! Flash Next's hyper-connection norm on 256 threads (tf_fn_hc_norm_v4) against the 1,024-thread kernel it replaces.

const std = @import("std");
const rig = @import("rig.zig");
const util = @import("../launches/util.zig");
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
