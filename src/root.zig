const std = @import("std");
const testing = std.testing;

// Neurona structure
const Neurona = struct {
    pesos: []f32,
    sesgo: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_entradas: usize) !Neurona {
        const pesos = try allocator.alloc(f32, num_entradas);
        return Neurona{
            .pesos = pesos,
            .sesgo = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Neurona) void {
        self.allocator.free(self.pesos);
    }
};

// Capa (Layer) structure
const Capa = struct {
    neuronas: []Neurona,
    funcion_activacion: *const fn (f32) f32,
    allocator: std.mem.Allocator,

    pub fn init(
            allocator: std.mem.Allocator
            , num_neuronas: usize
            , num_entradas: usize
            , funcion_activacion: *const fn (f32) f32
        ) !Capa {
        
        const neuronas = try allocator.alloc(Neurona, num_neuronas);
        errdefer allocator.free(neuronas);

        for (neuronas) |*neurona| {
            neurona.* = try Neurona.init(allocator, num_entradas);
        }

        return Capa{
            .neuronas = neuronas,
            .funcion_activacion = funcion_activacion,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Capa) void {
        for (self.neuronas) |*neurona| {
            neurona.deinit();
        }
        self.allocator.free(self.neuronas);
    }
};

// Weighted sum function
fn suma_ponderada(neurona: *const Neurona, entradas: []const f32) f32 {
    if (entradas.len != neurona.pesos.len) {
        @panic("El número de entradas no coincide con el número de pesos");
    }

    var suma: f32 = neurona.sesgo;
    for (neurona.pesos, 0..) |peso, i| {
        suma += peso * entradas[i];
    }
    return suma;
}

// Activation function: Sigmoid
fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}




////////////// STARTS UNIT TESTING

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

test "Capa initialization and deinitialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const num_neuronas: usize = 2;
    const num_entradas: usize = 3;

    var capa = try Capa.init(allocator, num_neuronas, num_entradas, &sigmoid);
    defer capa.deinit();

    // Check layer properties
    try std.testing.expectEqual(capa.neuronas.len, num_neuronas);
    try std.testing.expectEqual(capa.funcion_activacion, &sigmoid);

    // Check each neuron in the layer
    for (capa.neuronas) |neurona| {
        try std.testing.expectEqual(neurona.pesos.len, num_entradas);
        try std.testing.expectEqual(neurona.sesgo, 0);

        // Check that weights are initialized (they should not all be zero)
        var all_zero = true;
        for (neurona.pesos) |peso| {
            if (peso != 0) {
                all_zero = false;
                break;
            }
        }
        try std.testing.expect(!all_zero);
    }

    // The deinitialization will be called by the defer statement
}
