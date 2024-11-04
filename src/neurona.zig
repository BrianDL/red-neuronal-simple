const std = @import("std");


pub const Neurona = struct {
    pesos: []f32,
    sesgo: f32,
    allocator: std.mem.Allocator,

    pub fn init(
            allocator: std.mem.Allocator,
            num_entradas: usize,
            pesos: ?[]const f32,
            sesgo: ?f32
    ) !Neurona {
        const new_pesos = try allocator.alloc(f32, num_entradas);
        errdefer allocator.free(new_pesos);

        if (pesos) |p| {
            std.mem.copyForwards(f32, new_pesos, p);
        } else {
            @memset(new_pesos, 0.5);
        }

        return Neurona{
            .pesos = new_pesos,
            .sesgo = sesgo orelse 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Neurona) void {
        self.allocator.free(self.pesos);
    }
};

/////////////////// START UNIT TESTING

test "Neurona initialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var neurona = try Neurona.init(allocator, 3, null, 10);
    defer neurona.deinit();

    try std.testing.expectEqual(neurona.pesos.len, 3);
    try std.testing.expectEqual(neurona.sesgo, 10);
}