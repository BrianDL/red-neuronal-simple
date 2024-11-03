const std = @import("std");
const main = @import("src/main.zig");

const Neurona = main.Neurona;
const sigmoid = main.sigmoid;

test "Neurona initialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var neurona = try Neurona.init(allocator, 3);
    defer neurona.deinit();

    try std.testing.expectEqual(neurona.pesos.len, 3);
    try std.testing.expectEqual(neurona.sesgo, 0);
}

test "Sigmoid function" {
    const result = sigmoid(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result, 0.0001);
}