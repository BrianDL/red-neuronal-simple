const std = @import("std");

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

// Activation function: Sigmoid
fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

// Capa (Layer) structure
const Capa = struct {
    neuronas: []Neurona,
    funcion_activacion: fn (f32) f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_neuronas: usize, num_entradas: usize, funcion_activacion: fn (f32) f32) !Capa {
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

pub fn main() !void {
    // Initialize the allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a sample neuron
    var neurona = try Neurona.init(allocator, 3);
    defer neurona.deinit();

    // Initialize weights and bias
    neurona.pesos[0] = 0.5;
    neurona.pesos[1] = -0.3;
    neurona.pesos[2] = 0.8;
    neurona.sesgo = 0.1;

    // Sample inputs
    const entradas = [_]f32{ 1.0, 2.0, 3.0 };

    // Calculate weighted sum
    const suma = suma_ponderada(&neurona, &entradas);

    // Apply activation function
    const salida = sigmoid(suma);

    // Print results
    std.debug.print("Suma ponderada: {d:.4}\n", .{suma});
    std.debug.print("Salida de la neurona: {d:.4}\n", .{salida});

    // Create a sample layer
    var capa = try Capa.init(allocator, 2, 3, sigmoid);
    defer capa.deinit();

    std.debug.print("Capa creada con éxito\n", .{});
}

////// STARTS UNIT TESTING

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

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
