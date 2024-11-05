const std = @import("std");
const root = @import("root.zig");

const RedNeuronal = root.RedNeuronal;

const capa_module = @import("capa.zig");
const WeightInitStrategy = capa_module.WeightInitStrategy;
const linear = capa_module.linear;

const log = std.log;
pub const log_level: log.Level = .err;

pub fn main() !void {
    // Inicializar el asignador de memoria
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Configuración: 1 neurona de entrada, 1 neurona de salida
    const configuracion = [_]usize{ 1, 1 };
    const funciones_activacion = [_]*const fn (f32) f32{linear};

    // Crear la red neuronal
    var red = try RedNeuronal.init(allocator, &configuracion, &funciones_activacion, WeightInitStrategy.UniformRandom, null);
    defer red.deinit();

    // Datos de entrenamiento: y = 2x + 1
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

    // Crear slices para entradas y objetivos
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

    // Entrenar la red
    std.debug.print("Entrenando la red...\n", .{});
    try red.entrenar_simple(entradas, objetivos, 1000, 0.01);

    // Probar la red entrenada
    std.debug.print("\nProbando la red entrenada:\n", .{});
    for (entradas_raw, objetivos_raw) |entrada, objetivo| {
        const salida = try red.propagar_adelante(&entrada);
        defer allocator.free(salida);
        std.debug.print("Entrada: {d:.1}, Salida: {d:.4}, Objetivo: {d:.1}\n", .{ entrada[0], salida[0], objetivo[0] });
    }

    // Probar con una nueva entrada
    const nueva_entrada = [_]f32{5};
    const nueva_salida = try red.propagar_adelante(&nueva_entrada);
    defer allocator.free(nueva_salida);
    std.debug.print("\nNueva entrada: {d:.1}, Salida predicha: {d:.4}\n", .{ nueva_entrada[0], nueva_salida[0] });
}