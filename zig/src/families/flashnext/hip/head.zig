//! The MTP head on the drafting rank: its forward over (residual streams, next token) rows, and its draft chains.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;

pub const Out = forward.Out;

/// The head's caches: one sparse layer at the whole model's heads.
pub fn caches(
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    h: *const model.Head,
    cap: usize,
    act: model.Kind,
    eos: i64,
) !cache.Caches {
    return cache.Caches.init(gpa, d, h.dims, &.{false}, cap, act, eos);
}

/// `tokens` (each after its row) over `streams` from `pos`; `prompt`: a chunk's rows, caches only, outputs unread.
pub fn run(
    ops: Ops,
    prompt: bool,
    m: *const model.Model,
    host: *forward.Host,
    c: *cache.Caches,
    streams: Tensor,
    tokens: []const u32,
    pos: usize,
    ones: hip.DeviceBuffer,
) !Out {
    var o = ops;
    o.window = !prompt;
    o.prefill = prompt;
    const w = &m.mtp.?;
    const d = w.dims;
    const rows = tokens.len;
    if (rows * d.streams * m.act.size() > ones.len) return error.HeadRowsPastOnes;
    const ids = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(ids);
    const at_pos = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(at_pos);
    for (ids, at_pos, tokens, 0..) |*i, *p, t, r| {
        if (t >= d.vocab) return error.InvalidToken; // embed reads the table at id unchecked
        i.* = @intCast(t);
        p.* = @intCast(pos + r);
    }
    const dev_ids = try host.put(o, std.mem.sliceAsBytes(ids));
    const dev_pos = try host.put(o, std.mem.sliceAsBytes(at_pos));
    const emb = try take(o, m.act, rows * d.hidden);
    try o.embedRows(m.embed, dev_ids, rows, emb);
    const en = try take(o, m.act, rows * d.hidden);
    try o.rms(emb, w.e_norm, en, rows, d.hidden, d.eps);
    const e = try o.project(en, w.fc_e, rows, false);
    const hn = try take(o, m.act, rows * d.wide());
    const waves = try take(o, .f32, rows * hip.Launcher.fn_wide_waves);
    try o.l.fn_rms_wide(
        @ptrFromInt(streams.ptr),
        @ptrFromInt(w.h_norm),
        @ptrFromInt(hn.ptr),
        @ptrFromInt(waves.ptr),
        @backingInt(m.act),
        @intCast(rows),
        @intCast(d.wide()),
        d.eps,
        o.stream,
    );
    const h = try o.project(hn, w.fc_h, rows * d.streams, false);
    const x = try take(o, m.act, rows * d.wide());
    const l = w.layer;
    // the embedding broadcast over the streams: h += round(e * 1) per stream, then the layer's first norm
    try forward.hcNorm(o, m, h, e, .{ .ptr = ones.base(), .kind = m.act }, x, l.ahc.scale, rows);
    const inject = try take(o, m.act, rows * d.streams);
    const mixed = try forward.hcPre(o, m, l.ahc, x, rows, inject);
    const seg = [_]forward.Seg{.{ .c = c, .tokens = tokens, .pos = pos }};
    c.pooled = @min(c.pooled, pos / d.index_ratio);
    if (prompt) {
        // a prompt's rows: only the layer's caches are read later (its first draw runs the last row as a window)
        try mixers.sparseKeys(o, m, d, l.mixer.sparse, &seg, 0, mixed, rows, dev_pos);
        c.pooled = (pos + rows) / d.index_ratio;
        return .{ .mixed = x, .streams = h };
    }
    const branch = try mixers.sparse(o, m, d, l.mixer.sparse, &seg, 0, mixed, rows, dev_pos, host, null);
    try forward.hcNorm(o, m, h, branch, inject, x, l.mhc.scale, rows);
    const mixed2 = try forward.hcPre(o, m, l.mhc, x, rows, inject);
    const y = try hip.moe.run(o, m.act, l.moe, mixed2, rows, prompt);
    try forward.hcNorm(o, m, h, y, inject, x, w.mix.scale, rows);
    c.pooled = (pos + rows) / d.index_ratio;
    return .{ .mixed = try forward.hcPre(o, m, w.mix, x, rows, null), .streams = h };
}

/// A sampled stream's draft: the target's keyed draw over the head's candidates, at its verify row's position.
pub const Keyed = struct { gpa: std.mem.Allocator, s: lanes.Sampling, position: u64, dtype: serve.sample.Dtype };

/// FZ_KEYED_DRAFTS=0: a sampled stream drafts the head's argmax, as before (Metal's switch, the same name).
pub fn keyedOn() bool {
    const v = std.c.getenv("FZ_KEYED_DRAFTS") orelse return true;
    return v[0] != '0';
}

/// Candidates a keyed draft reads, as the drawer's sampled row: top_k + MARGIN, or 0 (argmax) for a wide top_k.
fn candidates(s: lanes.Sampling, n: usize) usize {
    if (s.temperature <= 0 or s.top_k == 0) return 0;
    const k = @min(n, @as(usize, s.top_k) + serve.sample.MARGIN);
    return if (k > serve.draw.max_candidates) 0 else k;
}

/// Each row's draft token (argmax, or for one keyed row the target's keyed draw) and the head's argmax probability.
pub fn draws(
    o: Ops,
    m: *const model.Model,
    host: *forward.Host,
    s: hip.Stream,
    mixed: Tensor,
    rows: usize,
    ids: []const u32,
    out: []u32,
    probs: []f32,
    keyed: ?Keyed,
) !void {
    const w = &m.mtp.?;
    const logits = try o.project(mixed, w.draft, rows, false);
    const arg = try o.arena.of(i32, 2 * rows);
    try o.argmaxRows(logits, rows, w.draft.n, arg);
    try o.tokenProb(logits, rows, w.draft.n, arg, arg + rows * 4);
    const k: usize = if (keyed) |kd| (if (rows == 1) candidates(kd.s, w.draft.n) else 0) else 0;
    var cand: u64 = 0;
    if (k > 0) {
        const ks = [1]i32{@intCast(k)};
        const ks_dev = try host.put(o, std.mem.sliceAsBytes(&ks));
        cand = try o.arena.take(k * 6);
        try o.topkRows(logits, 1, w.draft.n, ks_dev, k, cand, cand + k * 4);
    }
    const at = std.mem.alignForward(usize, host.used, 16);
    const words_len = 2 * rows * 4;
    const bytes = host.staging.bytes[at..][0 .. words_len + k * 6];
    try hip.runtime.check(host.d.api.hipMemcpyDtoHAsync(bytes.ptr, @ptrFromInt(arg), words_len, s.handle));
    if (k > 0) try hip.runtime.check(host.d.api.hipMemcpyDtoHAsync(bytes[words_len..].ptr, @ptrFromInt(cand), k * 6, s.handle));
    try s.synchronize();
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(bytes[0..words_len])));
    for (out[0..rows], probs[0..rows], words[0..rows], words[rows..]) |*t, *p, i, f| {
        t.* = ids[i];
        p.* = @bitCast(f);
    }
    if (k > 0) {
        const kd = keyed.?;
        const cids = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(bytes[words_len..][0 .. k * 4])));
        const cvals = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(bytes[words_len + k * 4 ..][0 .. k * 2])));
        var values: [serve.draw.max_candidates]f64 = undefined;
        var toks: [serve.draw.max_candidates]u64 = undefined;
        for (0..k) |j| {
            values[j] = serve.sample.widen(cvals, kd.dtype, j);
            toks[j] = ids[@intCast(cids[j])];
        }
        out[0] = @intCast(try lanes.sampling.choose(kd.gpa, values[0..k], toks[0..k], kd.position, kd.s));
    }
}
