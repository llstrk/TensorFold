//! fold=on and fold=bf16 (opt-in): the fold tile against float64 with BF16 weights, its copies and tall plan bytewise.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const cases = @import("cases.zig");
const prod = @import("product.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Case = cases.Case;

const pick = 16;

fn dense(name: []const u8, rows: usize, n: usize, k: usize, bits: u8, group: u16) Case {
    return .{ .name = name, .path = .prefill, .rows = rows, .n = n, .k = k, .bits = bits, .group = group };
}

fn routed(name: []const u8, rows: usize, n: usize, k: usize, bits: u8, down: bool, item_rows: usize) Case {
    return .{ .name = name, .path = .prefill, .rows = rows, .n = n, .k = k, .bits = bits, .group = 64, .experts = 8, .slots = 4, .down = down, .item_rows = item_rows };
}

/// The fold widths on ragged rows and columns, every group, dense and routed (gate/up and down).
const shapes = [_]Case{
    dense("4-bit g64", 300, 1001, 2112, 4, 64),            dense("5-bit g64", 300, 1001, 2112, 5, 64),
    dense("4-bit g32", 129, 257, 2048, 4, 32),             dense("5-bit g128", 640, 520, 4096, 5, 128),
    routed("routed 4-bit", 300, 512, 2048, 4, false, 128), routed("routed 5-bit down", 200, 1024, 512, 5, true, 128),
};

/// The largest |y - ref| over the terms' magnitudes: BF16 weights (half a BF16 unit a term) and an fp32 sum.
fn bound(k: usize) f64 {
    return 0x1p-9 + 0x1p-24 * (8 + 2 * @sqrt(@as(f64, @floatFromInt(k))));
}

fn at(i: usize, count: usize, n: usize) usize {
    return if (count <= 1 or n <= count) @min(i, n - 1) else i * (n - 1) / (count - 1);
}

/// The worst sampled output of `out` (fp32) over the fold bound.
fn worstOf(p: *const Product, out: Buffer) !f64 {
    const host = try gpa.alloc(f32, p.out_rows * p.c.n);
    defer gpa.free(host);
    try out.download(0, std.mem.sliceAsBytes(host));
    var worst: f64 = 0;
    for (0..@min(pick, p.out_rows)) |i| for (0..@min(pick, p.c.n)) |j| {
        const row = at(i, pick, p.out_rows);
        const col = at(j, pick, p.c.n);
        const v = p.reference(row, col);
        const y: f64 = host[row * p.c.n + col];
        worst = @max(worst, @abs(y - v.y) / (bound(p.c.k) * @max(v.norm, 1e-30)));
    };
    return worst;
}

/// Every prompt entry that takes `p`, launched into `out`; the digest of their bytes in order.
fn launchAll(p: *const Product, out: Buffer, worst: ?*f64) !u64 {
    var h: u64 = 0;
    var ran: usize = 0;
    for (p.rig.kernels.reg.entries) |*e| {
        if (e.path != .prefill or !prod.takes(p, e)) continue;
        try out.fill8(0xA5, null);
        try prod.launchEntry(p, e, out, false, 0, 0);
        try p.rig.base.stream.synchronize();
        if (worst) |w| w.* = @max(w.*, try worstOf(p, out));
        const d = try prod.digestOf(out, p.outBytes(false));
        h = std.hash.Wyhash.hash(h, std.mem.asBytes(&d));
        ran += 1;
    }
    try base.expect(ran > 0, "fold {s}: no prompt entry takes the product", .{p.c.name});
    return h;
}

fn open() !?*base.Rig {
    const b = try base.Rig.open(1 << 20);
    if (!b.on.affine.matrix or !b.bf16) {
        b.close();
        return null;
    }
    return b;
}

test "fold=on: the prompt matrix tile sits within BF16 weights' rounding of float64, and writes other bytes than off" {
    const b = try open() orelse return error.SkipZigTest;
    defer b.close();
    var rig = Rig.open(b);
    var worst: f64 = 0;
    for (shapes) |c| {
        var p = try Product.init(&rig, c);
        defer p.deinit();
        var out = try Buffer.alloc(b, p.outBytes(false));
        defer out.free();
        b.on.affine.matrix_fold = false;
        const off = try launchAll(&p, out, null);
        b.on.affine.matrix_fold = true;
        defer b.on.affine.matrix_fold = false;
        var w: f64 = 0;
        const on = try launchAll(&p, out, &w);
        try base.expect(w <= 1, "fold {s}: {d:.2} times the BF16-weight bound against float64", .{ c.name, w });
        try base.expect(on != off, "fold {s}: fold=on wrote the bytes of fold off (the fold tile did not run)", .{c.name});
        worst = @max(worst, w);
    }
    std.debug.print("FOLD {d} products within {d:.2} of the BF16-weight bound of float64\n", .{ shapes.len, worst });
}

test "fold=bf16: a dense product on its BF16 copy writes fold=on's bytes" {
    const b = try open() orelse return error.SkipZigTest;
    defer b.close();
    var rig = Rig.open(b);
    b.on.affine.matrix_fold = true;
    defer b.on.affine.matrix_fold = false;
    var made: usize = 0;
    for ([_]Case{ dense("copy 4-bit", 600, 1001, 2112, 4, 64), dense("copy 5-bit", 512, 520, 4096, 5, 128), dense("copy 4-bit g32", 1024, 384, 1024, 4, 32) }) |c| {
        var p = try Product.init(&rig, c);
        defer p.deinit();
        var out = try Buffer.alloc(b, p.outBytes(false));
        defer out.free();
        const plain = try launchAll(&p, out, null);
        var copies: affine.Copies = .{ .gpa = gpa, .limit = 1 << 30 };
        defer copies.deinit();
        b.on.affine.copies = &copies;
        defer b.on.affine.copies = null;
        // twice: the launch that makes the copy, and one that finds it made
        for (0..2) |_| try base.expect(try launchAll(&p, out, null) == plain, "fold=bf16 {s}: the copy's tile writes other bytes than fold=on", .{c.name});
        try base.expect(copies.map.count() == 1, "fold=bf16 {s}: {d} copies made, not 1", .{ c.name, copies.map.count() });
        made += 1;
    }
    std.debug.print("FOLD16 {d} products on their copies write fold=on's bytes\n", .{made});
}

test "fold=on: the tall plan's 256-row items write the 128-row plan's bytes, gate/up and down" {
    const b = try open() orelse return error.SkipZigTest;
    defer b.close();
    var rig = Rig.open(b);
    b.on.affine.matrix_fold = true;
    defer b.on.affine.matrix_fold = false;
    var compared: usize = 0;
    for ([_]bool{ false, true }) |down| {
        rig.rng = .{ .state = 0x7A11 };
        var short = try Product.init(&rig, routed("tall", 600, 512, 1024, 4, down, 128));
        defer short.deinit();
        rig.rng = .{ .state = 0x7A11 };
        var tall = try Product.init(&rig, routed("tall", 600, 512, 1024, 4, down, 256));
        defer tall.deinit();
        try base.expect(tall.max_rows > 128, "tall: no item past 128 rows ({d})", .{tall.max_rows});
        var out = try Buffer.alloc(b, short.outBytes(false));
        defer out.free();
        // the gemm and block tiles refuse the tall plan (the planner never gives it them), so they are skipped
        for (rig.kernels.reg.entries) |*e| {
            if (e.path != .prefill or !prod.takes(&short, e) or !prod.takes(&tall, e)) continue;
            if (std.mem.endsWith(u8, e.id, ".gemm") or std.mem.endsWith(u8, e.id, ".block")) continue;
            try out.fill8(0xA5, null);
            try prod.launchEntry(&tall, e, out, false, 0, 0);
            try b.stream.synchronize();
            const w = try worstOf(&tall, out);
            try base.expect(w <= 1, "tall {s}: {d:.2} times the BF16-weight bound against float64", .{ e.id, w });
            const got = try prod.digestOf(out, tall.outBytes(false));
            try out.fill8(0xA5, null);
            try prod.launchEntry(&short, e, out, false, 0, 0);
            try b.stream.synchronize();
            try base.expect(got == try prod.digestOf(out, short.outBytes(false)), "tall (down {}): {s} on the 256-row plan writes other bytes than on the 128-row one", .{ down, e.id });
            compared += 1;
        }
    }
    try base.expect(compared >= 2, "tall: only {d} launches took the tall plan", .{compared});
    std.debug.print("TALL {d} launches on the 256-row plan write the 128-row plan's bytes\n", .{compared});
}
