const std = @import("std");
const testing = std.testing;

const Neurona = @import("neurona.zig").Neurona;

const capa_module = @import("capa.zig");
const Capa = capa_module.Capa;
const sigmoid = capa_module.sigmoid;
const WeightInitStrategy = capa_module.WeightInitStrategy;

const RedNeuronal = struct {
    capas: []Capa,
    allocator: std.mem.Allocator,
    strat_inicia_pesos: WeightInitStrategy,

    pub fn init(allocator: std.mem.Allocator, configuracion: []const usize, funciones_activacion: []const *const fn (f32) f32, strat_inicia_pesos: WeightInitStrategy, semilla: ?u64) !RedNeuronal {
        const capas = try allocator.alloc(Capa, configuracion.len - 1);
        errdefer allocator.free(capas);

        for (capas, 0..) |*capa, i| {
            capa.* = try Capa.init(allocator, configuracion[i + 1], configuracion[i], funciones_activacion[i], strat_inicia_pesos, if (semilla) |s| s +% i else null);
        }

        return RedNeuronal{ .capas = capas, .allocator = allocator, .strat_inicia_pesos = strat_inicia_pesos };
    }

    pub fn deinit(self: *RedNeuronal) void {
        for (self.capas) |*capa| {
            capa.deinit();
        }
        self.allocator.free(self.capas);
    }

    pub fn propagar_adelante(self: *const RedNeuronal, entradas: []const f32) ![]f32 {
        var salida_actual = try self.allocator.dupe(f32, entradas);
        defer self.allocator.free(salida_actual);

        std.debug.print("Input: ", .{});
        for (salida_actual) |valor| {
            std.debug.print("{d:.6} ", .{valor});
        }
        std.debug.print("\n", .{});

        for (self.capas, 0..) |capa, i| {
            var salida_capa = try self.allocator.alloc(f32, capa.neuronas.len);
            defer self.allocator.free(salida_capa);

            std.debug.print("Layer {d}:\n", .{i + 1});
            for (capa.neuronas, 0..) |neurona, j| {
                const suma = suma_ponderada(&neurona, salida_actual);
                std.debug.print("  Neuron {d} sum: {d:.6}\n", .{ j + 1, suma });
                salida_capa[j] = capa.funcion_activacion(suma);
                std.debug.print("  Neuron {d} output: {d:.6}\n", .{ j + 1, salida_capa[j] });
            }

            self.allocator.free(salida_actual);
            salida_actual = try self.allocator.dupe(f32, salida_capa);

            std.debug.print("Layer {d} output: ", .{i + 1});
            for (salida_actual) |valor| {
                std.debug.print("{d:.6} ", .{valor});
            }
            std.debug.print("\n", .{});
        }

        // Create a new slice for the final output
        const final_output = try self.allocator.dupe(f32, salida_actual);
        // Debug: Print final output before returning
        std.debug.print("Final output before return: ", .{});
        for (final_output) |valor| {
            std.debug.print("{d:.6} ", .{valor});
        }
        std.debug.print("\n", .{});

        return final_output;
    }

    pub fn entrenar_simple(self: *RedNeuronal, entradas: []const []const f32, objetivos: []const []const f32, epocas: usize, tasa_aprendizaje: f32) !void {
        for (0..epocas) |epoca| {
            var perdida_total: f32 = 0;
            for (entradas, objetivos) |entrada, objetivo| {
                const prediccion = try self.propagar_adelante(entrada);
                defer self.allocator.free(prediccion);

                const perdida = mse(prediccion, objetivo);
                perdida_total += perdida;

                // Ajuste simple de pesos y sesgos
                for (self.capas) |*capa| {
                    for (capa.neuronas) |*neurona| {
                        for (neurona.pesos, 0..) |*peso, j| {
                            peso.* += tasa_aprendizaje * (objetivo[0] - prediccion[0]) * entrada[j];
                        }
                        neurona.sesgo += tasa_aprendizaje * (objetivo[0] - prediccion[0]);
                    }
                }
            }
            std.debug.print("Época {}: Pérdida promedio = {d:.6}\n", .{ epoca + 1, perdida_total / @intToFloat(f32, entradas.len) });
        }
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

fn mse(predicciones: []const f32, objetivos: []const f32) f32 {
    if (predicciones.len != objetivos.len) {
        @panic("Las longitudes de predicciones y objetivos no coinciden");
    }
    var suma_errores: f32 = 0;
    for (predicciones, objetivos) |pred, obj| {
        const err = pred - obj;
        suma_errores += err * err;
    }

    return suma_errores / @as(f32, @floatFromInt(predicciones.len));
}

////////////// STARTS UNIT TESTING

test "Mean Squared Error (MSE) calculation" {
    const epsilon = 0.000001;

    // Test case 1: Perfect prediction
    {
        const predicciones = [_]f32{ 1.0, 2.0, 3.0 };
        const objetivos = [_]f32{ 1.0, 2.0, 3.0 };
        const result = mse(&predicciones, &objetivos);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, epsilon);
    }

    // Test case 2: Some error
    {
        const predicciones = [_]f32{ 1.0, 2.0, 3.0 };
        const objetivos = [_]f32{ 1.1, 2.2, 2.8 };
        const result = mse(&predicciones, &objetivos);
        try std.testing.expectApproxEqAbs(@as(f32, 0.03), result, epsilon);
    }

    // Test case 3: Large error
    {
        const predicciones = [_]f32{ 0.0, 0.0, 0.0 };
        const objetivos = [_]f32{ 1.0, 1.0, 1.0 };
        const result = mse(&predicciones, &objetivos);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, epsilon);
    }

    // Test case 4: Mixed positive and negative errors
    {
        const predicciones = [_]f32{ 0.5, 1.5, 2.5 };
        const objetivos = [_]f32{ 1.0, 1.0, 3.0 };
        const result = mse(&predicciones, &objetivos);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), result, epsilon);
    }
}

test "RedNeuronal initialization and forward propagation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Configuration: 2 input neurons, 1 output neuron
    const configuracion = [_]usize{ 2, 1 };

    // Only one activation function for the output layer
    const funciones_activacion = [_]*const fn (f32) f32{sigmoid};

    var red = try RedNeuronal.init(allocator, &configuracion, &funciones_activacion, WeightInitStrategy.Constant, null);
    defer red.deinit();

    // Check the network structure
    try std.testing.expectEqual(red.capas.len, 1);
    try std.testing.expectEqual(red.capas[0].neuronas.len, 1);

    // Print weights and biases
    std.debug.print("Weights: ", .{});
    for (red.capas[0].neuronas[0].pesos) |peso| {
        std.debug.print("{d:.6} ", .{peso});
    }
    std.debug.print("\nBias: {d:.6}\n", .{red.capas[0].neuronas[0].sesgo});

    // Test forward propagation
    const entradas = [_]f32{ 0.5, 0.8 };
    const salida = try red.propagar_adelante(&entradas);
    defer allocator.free(salida);

    std.debug.print("Final Output: {d:.6}\n", .{salida[0]});

    // Check the output
    try std.testing.expectEqual(salida.len, 1);
    try std.testing.expect(salida[0] > 0 and salida[0] < 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.657010), salida[0], 0.000001);
}
