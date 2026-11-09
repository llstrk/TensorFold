//! The linear attention's prompt conv with its q, k and v channels split into three contiguous outputs.

const std = @import("std");
const rig = @import("rig.zig");
const ops = @import("../ops/ops.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

test "the prompt conv's split q, k and v rows hold the bytes of its single output's columns" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xC0418 };
    // the key width cuts the first 256-channel block, and the values end inside the second
    const len = 37;
    const kw = 96;
    const vw = 172;
    const channels = 2 * kw + vw;
    const kernel = 4;
    const kept = kernel - 1;
    const hx = try gpa.alloc(u16, len * channels);
    defer gpa.free(hx);
    for (hx) |*x| x.* = t.bits(rng.unit() * 2);
    const hw = try gpa.alloc(f32, channels * kernel);
    defer gpa.free(hw);
    for (hw) |*x| x.* = rng.unit() / 2;
    const hs = try gpa.alloc(f32, kept * channels);
    defer gpa.free(hs);
    for (hs) |*x| x.* = rng.unit();
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var state = try t.upload(hs);
    defer state.free();
    var out = try t.alloc(len * channels * 4);
    defer out.free();
    var next = try t.alloc(kept * channels * 4);
    defer next.free();
    var q = try t.alloc(len * kw * 4);
    defer q.free();
    var k = try t.alloc(len * kw * 4);
    defer k.free();
    var v = try t.alloc(len * vw * 4);
    defer v.free();
    var split_next = try t.alloc(kept * channels * 4);
    defer split_next.free();
    const xs: ops.Tensor = .{ .ptr = at(x), .kind = if (t.bf16) .bf16 else .f16 };
    try o.convPrefill(xs, at(w), at(state), at(out), at(next), len, channels, kernel, null);
    const split: ops.ConvSplit = .{ .k = at(k), .v = at(v), .key_width = kw };
    try o.convPrefill(xs, at(w), at(state), at(q), at(split_next), len, channels, kernel, split);
    try t.stream.synchronize();
    const go = try rig.download(u32, out);
    defer gpa.free(go);
    const gq = try rig.download(u32, q);
    defer gpa.free(gq);
    const gk = try rig.download(u32, k);
    defer gpa.free(gk);
    const gv = try rig.download(u32, v);
    defer gpa.free(gv);
    for (0..len) |r| {
        const row = go[r * channels ..][0..channels];
        try std.testing.expectEqualSlices(u32, row[0..kw], gq[r * kw ..][0..kw]);
        try std.testing.expectEqualSlices(u32, row[kw .. 2 * kw], gk[r * kw ..][0..kw]);
        try std.testing.expectEqualSlices(u32, row[2 * kw ..], gv[r * vw ..][0..vw]);
    }
    const gn = try rig.download(u32, next);
    defer gpa.free(gn);
    const gs = try rig.download(u32, split_next);
    defer gpa.free(gs);
    try std.testing.expectEqualSlices(u32, gn, gs);
}

test "the prompt conv's normed q and k hold the bytes of the split conv's rows after tf_rms_rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xC0419 };
    // two 128-wide heads a 256-channel block; 37 rows end inside the third 16-row block
    const len = 37;
    const kw = 512;
    const vw = 256;
    const channels = 2 * kw + vw;
    const kernel = 4;
    const kept = kernel - 1;
    const width = 128;
    const eps: f32 = 1e-6;
    const hx = try gpa.alloc(u16, len * channels);
    defer gpa.free(hx);
    for (hx) |*x| x.* = t.bits(rng.unit() * 2);
    const hw = try gpa.alloc(f32, channels * kernel);
    defer gpa.free(hw);
    for (hw) |*x| x.* = rng.unit() / 2;
    const hs = try gpa.alloc(f32, kept * channels);
    defer gpa.free(hs);
    for (hs) |*x| x.* = rng.unit();
    var hn: [2][width]f32 = undefined;
    for (&hn) |*row| for (row) |*x| {
        x.* = 1 + rng.unit() / 4;
    };
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var state = try t.upload(hs);
    defer state.free();
    var qn = try t.upload(&hn[0]);
    defer qn.free();
    var kn = try t.upload(&hn[1]);
    defer kn.free();
    var bufs: [10]DeviceBuffer = undefined;
    const sizes = [_]usize{ len * kw, len * kw, len * vw, kept * channels, len * kw, len * kw, len * kw, len * kw, len * vw, kept * channels };
    for (&bufs, sizes) |*b, n| b.* = try t.alloc(n * 4);
    defer for (&bufs) |*b| b.free();
    const qc, const kc, const v0, const next0, const q0, const k0, const q1, const k1, const v1, const next1 = bufs;
    const xs: ops.Tensor = .{ .ptr = at(x), .kind = if (t.bf16) .bf16 else .f16 };
    for ([_]bool{ true, false }) |weighted| {
        const qw: ?u64 = if (weighted) at(qn) else null;
        const kwt: ?u64 = if (weighted) at(kn) else null;
        try o.convPrefill(xs, at(w), at(state), at(qc), at(next0), len, channels, kernel, .{ .k = at(kc), .v = at(v0), .key_width = kw });
        try o.rms(.{ .ptr = at(qc), .kind = .f32 }, qw, .{ .ptr = at(q0), .kind = .f32 }, len * kw / width, width, eps);
        try o.rms(.{ .ptr = at(kc), .kind = .f32 }, kwt, .{ .ptr = at(k0), .kind = .f32 }, len * kw / width, width, eps);
        const norm: ops.QkNorm = .{ .q = qw, .k = kwt, .eps = eps };
        try o.convPrefill(xs, at(w), at(state), at(q1), at(next1), len, channels, kernel, .{ .k = at(k1), .v = at(v1), .key_width = kw, .norm = norm });
        try t.stream.synchronize();
        for ([_][2]DeviceBuffer{ .{ q0, q1 }, .{ k0, k1 }, .{ v0, v1 }, .{ next0, next1 } }) |pair| {
            const want = try rig.download(u32, pair[0]);
            defer gpa.free(want);
            const got = try rig.download(u32, pair[1]);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}
