const std = @import("std");
const root = @import("root.zig");

const RedNeuronal = root.RedNeuronal;

const capa_module = @import("capa.zig");
const WeightInitStrategy = capa_module.WeightInitStrategy;
const linear = capa_module.linear;

const log = std.log;
pub const log_level: log.Level = .info;

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

    // Datos de entrenamiento: y = 3x -10
    const entradas_slice = &[_][]const f32{
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{2},
        &[_]f32{3},
        &[_]f32{4},
    };

    const objetivos_slice = &[_][]const f32{
        &[_]f32{-10}, // 3*0 -10
        &[_]f32{-7}, // 2*1 + 1
        &[_]f32{-4}, // 2*2 + 1
        &[_]f32{-1}, // 2*3 + 1
        &[_]f32{2}, // 2*4 + 1
    };

    // Imprimir los contenidos de entradas_slice y objetivos_slice
    log.info("Contenido de entradas_slice:", .{});
    for (entradas_slice, 0..) |entrada, i| {
        log.info("  Entrada {d}: {d}", .{ i, entrada[0] });
    }

    log.info("Contenido de objetivos_slice:", .{});
    for (objetivos_slice, 0..) |objetivo, i| {
        log.info("  Objetivo {d}: {d}", .{ i, objetivo[0] });
    }

    // Entrenar la red
    log.info("Entrenando la red...", .{});
    try red.entrenar_simple(entradas_slice, objetivos_slice, 1000, 0.01);

    // Probar la red entrenada
    std.debug.print("\nProbando la red entrenada:\n", .{});
    for (entradas_slice, objetivos_slice) |entrada, objetivo| {
        const salida = try red.propagar_adelante(entrada);
        defer allocator.free(salida);
        std.debug.print("Entrada: {d:.1}, Salida: {d:.4}, Objetivo: {d:.1}\n", .{ entrada[0], salida[0], objetivo[0] });
    }

    // Probar con una nueva entrada
    const nueva_entrada = [_]f32{5};
    const nueva_salida = try red.propagar_adelante(&nueva_entrada);
    defer allocator.free(nueva_salida);
    std.debug.print("\nNueva entrada: {d:.1}, Salida predicha: {d:.4}\n", .{ nueva_entrada[0], nueva_salida[0] });
}