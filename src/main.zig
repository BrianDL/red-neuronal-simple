const std = @import("std");
const root = @import("root");

const sigmoid = root.sigmoid;

const Neurona = root.Neurona;
const Capa = root.Capa;

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
    const suma = root.suma_ponderada(&neurona, &entradas);

    // Apply activation function
    const salida = sigmoid(suma);

    // Print results
    std.debug.print("Suma ponderada: {d:.4}\n", .{suma});
    std.debug.print("Salida de la neurona: {d:.4}\n", .{salida});

    // Create a sample layer
    var capa = try Capa.init(allocator, 2, 3, &sigmoid);
    defer capa.deinit();

    std.debug.print("Capa creada con éxito\n", .{});
}