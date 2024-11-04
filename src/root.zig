const std = @import("std");
const testing = std.testing;

const Neurona = struct {
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

// Capa (Layer) structure
const Capa = struct {
    neuronas: []Neurona,
    funcion_activacion: *const fn (f32) f32,
    allocator: std.mem.Allocator,
    strat_inicia_pesos: WeightInitStrategy,

    pub fn init(
        allocator: std.mem.Allocator,
        num_neuronas: usize,
        num_entradas: usize,
        funcion_activacion: *const fn (f32) f32,
        strat_inicia_pesos: WeightInitStrategy,
        seed: ?u64
    ) !Capa {
        const neuronas = try allocator.alloc(Neurona, num_neuronas);
        errdefer allocator.free(neuronas);

        var prng = std.rand.DefaultPrng.init(
            seed orelse @intCast(std.time.milliTimestamp()));
        
        var rand = prng.random();

        for (neuronas) |*neurona| {
            neurona.* = try Neurona.init(allocator, num_entradas, null, 0);
            
            // Este bloque aplica la inicialización de pesos
            switch (strat_inicia_pesos) {
                .Constant => @memset(neurona.pesos, 0.5),
                .UniformRandom => for (neurona.pesos) |*peso| {
                    peso.* = rand.float(f32);
                },
                // .Xavier => {
                //     const limit = @sqrt(6.0 / @floatFromInt(num_entradas + num_neuronas));
                //     for (neurona.pesos) |*peso| {
                //         peso.* = rand.float(f32) * 2 * limit - limit;
                //     }
                // },
            }
        }

        return Capa{
            .neuronas = neuronas,
            .funcion_activacion = funcion_activacion,
            .allocator = allocator,
            .strat_inicia_pesos = strat_inicia_pesos,
        };

    }

    pub fn deinit(self: *Capa) void {
        for (self.neuronas) |*neurona| {
            neurona.deinit();
        }
        self.allocator.free(self.neuronas);
    }
};

const RedNeuronal = struct {
    capas: []Capa,
    allocator: std.mem.Allocator,
    strat_inicia_pesos: WeightInitStrategy,

    pub fn init(
        allocator: std.mem.Allocator,
        configuracion: []const usize,
        funciones_activacion: []const *const fn(f32) f32,
        strat_inicia_pesos: WeightInitStrategy,
        semilla: ?u64
    ) !RedNeuronal {
        const capas = try allocator.alloc(Capa, configuracion.len - 1);
        errdefer allocator.free(capas);

        for (capas, 0..) |*capa, i| {
            capa.* = try Capa.init(
                allocator,
                configuracion[i + 1],
                configuracion[i],
                funciones_activacion[i],
                strat_inicia_pesos,
                if (semilla) |s| s +% i else null
            );
        }

        return RedNeuronal{
            .capas = capas,
            .allocator = allocator,
            .strat_inicia_pesos = strat_inicia_pesos
        };
    }
    
    pub fn deinit(self: *RedNeuronal) void {
        for (self.capas) |*capa| {
            capa.deinit();
        }
        self.allocator.free(self.capas);
    }

    pub fn propagar_adelante(
            self: *const RedNeuronal,
            entradas: []const f32
        ) ![]f32 {

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
                std.debug.print("  Neuron {d} sum: {d:.6}\n", .{j + 1, suma});
                salida_capa[j] = capa.funcion_activacion(suma);
                std.debug.print("  Neuron {d} output: {d:.6}\n", .{j + 1, salida_capa[j]});
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
};

const WeightInitStrategy = enum {
    Constant,
    UniformRandom,
    // Xavier,
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

    var neurona = try Neurona.init(allocator, 3, null, 10);
    defer neurona.deinit();

    try std.testing.expectEqual(neurona.pesos.len, 3);
    try std.testing.expectEqual(neurona.sesgo, 10);
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

    // Test Constant initialization
    {
        var capa = try Capa.init(
            allocator,
            num_neuronas,
            num_entradas,
            &sigmoid,
            WeightInitStrategy.Constant,
            null
        );
        defer capa.deinit();

        try std.testing.expectEqual(capa.neuronas.len, num_neuronas);
        try std.testing.expectEqual(capa.funcion_activacion, &sigmoid);
        try std.testing.expectEqual(capa.strat_inicia_pesos, WeightInitStrategy.Constant);

        for (capa.neuronas) |neurona| {
            try std.testing.expectEqual(neurona.pesos.len, num_entradas);
            try std.testing.expectEqual(neurona.sesgo, 0);

            for (neurona.pesos) |peso| {
                try std.testing.expectApproxEqAbs(@as(f32, 0.5), peso, 0.0001);
            }
        }
    }

    // Test UniformRandom initialization
    {
        var capa = try Capa.init(
            allocator,
            num_neuronas,
            num_entradas,
            &sigmoid,
            WeightInitStrategy.UniformRandom,
            1234  // Use a fixed seed for reproducibility
        );
        defer capa.deinit();

        try std.testing.expectEqual(capa.neuronas.len, num_neuronas);
        try std.testing.expectEqual(capa.funcion_activacion, &sigmoid);
        try std.testing.expectEqual(capa.strat_inicia_pesos, WeightInitStrategy.UniformRandom);

        for (capa.neuronas) |neurona| {
            try std.testing.expectEqual(neurona.pesos.len, num_entradas);
            try std.testing.expectEqual(neurona.sesgo, 0);

            var all_same = true;
            const first_weight = neurona.pesos[0];
            for (neurona.pesos) |peso| {
                if (peso != first_weight) {
                    all_same = false;
                    break;
                }
                try std.testing.expect(peso >= 0 and peso < 1);
            }
            try std.testing.expect(!all_same);
        }
    }
}

test "RedNeuronal initialization and forward propagation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Configuration: 2 input neurons, 1 output neuron
    const configuracion = [_]usize{ 2, 1 };

    // Only one activation function for the output layer
    const funciones_activacion = [_]*const fn(f32) f32{sigmoid};

    var red = try RedNeuronal.init(
        allocator, &configuracion, &funciones_activacion
        , WeightInitStrategy.Constant, null
    );
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