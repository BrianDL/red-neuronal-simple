const std = @import("std");
const log = std.log;
const testing = std.testing;

// pub const log_level: log.Level = .err;

const Neurona = @import("neurona.zig").Neurona;

const capa_module = @import("capa.zig");
const Capa = capa_module.Capa;
const sigmoid = capa_module.sigmoid;
const WeightInitStrategy = capa_module.WeightInitStrategy;
const linear = capa_module.linear;

pub const RedNeuronal = struct {
    capas: []Capa,
    allocator: std.mem.Allocator,
    strat_inicia_pesos: WeightInitStrategy,

    pub fn init(allocator: std.mem.Allocator
            , configuracion: []const usize
            , funciones_activacion: []const *const fn (f32) f32
            , strat_inicia_pesos: WeightInitStrategy
            , semilla: ?u64
        ) !RedNeuronal {

        const capas = try allocator.alloc(Capa, configuracion.len - 1);
        errdefer allocator.free(capas);

        for (capas, 0..) |*capa, i| {
            capa.* = try Capa.init(
                allocator
                , configuracion[i + 1]
                , configuracion[i]
                , funciones_activacion[i]
                , strat_inicia_pesos
                , if (semilla) |s| s +% i else null);
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

        // log.debug("Input: {any}", .{salida_actual});

        for (self.capas) |capa| {
            var salida_capa = try self.allocator.alloc(f32, capa.neuronas.len);
            defer self.allocator.free(salida_capa);

            // log.debug("Layer {d}:", .{i + 1});
            for (capa.neuronas, 0..) |neurona, j| {
                const suma = suma_ponderada(&neurona, salida_actual);
                // log.debug("  Neuron {d} sum: {d:.6}", .{ j + 1, suma });
                salida_capa[j] = capa.funcion_activacion(suma);
                // log.debug("  Neuron {d} output: {d:.6}", .{ j + 1, salida_capa[j] });
            }

            self.allocator.free(salida_actual);
            salida_actual = try self.allocator.dupe(f32, salida_capa);

            // log.debug("Layer {d} output: {any}", .{ i + 1, salida_actual });
        }

        // Create a new slice for the final output
        const final_output = try self.allocator.dupe(f32, salida_actual);
        // log.debug("Final output: {any}", .{final_output});

        return final_output;
    }

    pub fn entrenar_simple(self: *RedNeuronal
            , entradas: []const []const f32
            , objetivos: []const []const f32
            , epocas: usize
            , tasa_aprendizaje: f32
        ) !void {
        
        for (0..epocas) |_| {
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
            // log.debug("Época {}: Pérdida promedio = {d:.6}\n", .{ epoca + 1, perdida_total / @as(f32, @floatFromInt(entradas.len)) });
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

    // // Log weights and biases
    // // log.debug("Weights: ", .{});
    // for (red.capas[0].neuronas[0].pesos) |peso| {
    //     // log.debug("{d:.6} ", .{peso});
    // }
    // // log.debug("Bias: {d:.6}", .{red.capas[0].neuronas[0].sesgo});

    // Test forward propagation
    const entradas = [_]f32{ 0.5, 0.8 };
    const salida = try red.propagar_adelante(&entradas);
    defer allocator.free(salida);

    // log.debug("Final Output: {d:.6}", .{salida[0]});

    // Check the output
    try std.testing.expectEqual(salida.len, 1);
    try std.testing.expect(salida[0] > 0 and salida[0] < 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.657010), salida[0], 0.000001);
}

test "Simple training of RedNeuronal - Linear Regression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const configuracion = [_]usize{ 1, 1 }; // 1 input, 1 output
    const funciones_activacion = [_]*const fn (f32) f32{linear};

    var red = try RedNeuronal.init(allocator, &configuracion, &funciones_activacion, WeightInitStrategy.UniformRandom, null);
    defer red.deinit();

    const entradas_raw = [_][1]f32{
        .{0},
        .{1},
        .{2},
        .{3},
        .{4},
    };
    const objetivos_raw = [_][1]f32{
        .{1}, // 2*0 + 1
        .{3}, // 2*1 + 1
        .{5}, // 2*2 + 1
        .{7}, // 2*3 + 1
        .{9}, // 2*4 + 1
    };

    // Create slices directly
    const entradas: []const []const f32 = &[_][]const f32{
        &entradas_raw[0],
        &entradas_raw[1],
        &entradas_raw[2],
        &entradas_raw[3],
        &entradas_raw[4],
    };

    const objetivos: []const []const f32 = &[_][]const f32{
        &objetivos_raw[0],
        &objetivos_raw[1],
        &objetivos_raw[2],
        &objetivos_raw[3],
        &objetivos_raw[4],
    };

    // // Debug log entradas and objetivos
    // // log.debug("Entradas:", .{});
    // for (entradas) |entrada| {
    //     // log.debug("{any}", .{entrada});
    // }
    // // log.debug("Objetivos:", .{});
    // for (objetivos) |objetivo| {
    //     // log.debug("{any}", .{objetivo});
    // }

    try red.entrenar_simple(entradas, objetivos, 100, 0.1);

    // Test the trained network
    for (entradas_raw, objetivos_raw) |entrada, objetivo| {
        const salida = try red.propagar_adelante(&entrada);
        defer allocator.free(salida);
        // log.debug("Input: {d:.1}, Output: {d:.4}, Target: {d:.1}", .{ entrada[0], salida[0], objetivo[0] });
        try std.testing.expectApproxEqAbs(objetivo[0], salida[0], 0.5);
    }
}
