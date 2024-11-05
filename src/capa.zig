const std = @import("std");

const Neurona = @import("neurona.zig").Neurona;

// Capa (Layer) structure
pub const Capa = struct {
    neuronas: []Neurona,
    funcion_activacion: *const fn (f32) f32,
    allocator: std.mem.Allocator,
    strat_inicia_pesos: WeightInitStrategy,

    pub fn init(
            allocator: std.mem.Allocator
            , num_neuronas: usize
            , num_entradas: usize
            , funcion_activacion: *const fn (f32) f32
            , strat_inicia_pesos: WeightInitStrategy
            , seed: ?u64
        ) !Capa {
            
        const neuronas = try allocator.alloc(Neurona, num_neuronas);
        errdefer allocator.free(neuronas);

        var prng = std.rand.DefaultPrng.init(seed orelse @intCast(std.time.milliTimestamp()));

        var rand = prng.random();

        for (neuronas) |*neurona| {
            neurona.* = try Neurona.init(allocator, num_entradas, null, 0);

            // Este bloque aplica la inicialización de pesos
            switch (strat_inicia_pesos) {
                .Constant => @memset(neurona.pesos, 0.5),
                .UniformRandom => for (neurona.pesos) |*peso| {
                    peso.* = rand.float(f32);
                },
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

pub const WeightInitStrategy = enum {
    Constant,
    UniformRandom,
};

// Activation function: Sigmoid
pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

// Activation function: Linear
pub fn linear(x: f32) f32 {
    return x;
}

/////////////////  START UNIT TESTING

test "Sigmoid function" {
    const result = sigmoid(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result, 0.0001);
}

test "Linear function" {
    try std.testing.expectEqual(@as(f32, -1.0), linear(-1.0));
    try std.testing.expectEqual(@as(f32, 0.0), linear(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), linear(1.0));
    try std.testing.expectEqual(@as(f32, 100.0), linear(100.0));
}
test "Capa initialization and deinitialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const num_neuronas: usize = 2;
    const num_entradas: usize = 3;

    // Test Constant initialization with sigmoid
    {
        var capa = try Capa.init(allocator, num_neuronas, num_entradas, &sigmoid, WeightInitStrategy.Constant, null);
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

    // Test UniformRandom initialization with sigmoid
    {
        var capa = try Capa.init(allocator, num_neuronas, num_entradas, &sigmoid, WeightInitStrategy.UniformRandom, 1234);
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

    // Test Constant initialization with linear activation
    {
        var capa = try Capa.init(allocator, num_neuronas, num_entradas, &linear, WeightInitStrategy.Constant, null);
        defer capa.deinit();

        try std.testing.expectEqual(capa.neuronas.len, num_neuronas);
        try std.testing.expectEqual(capa.funcion_activacion, &linear);
        try std.testing.expectEqual(capa.strat_inicia_pesos, WeightInitStrategy.Constant);

        for (capa.neuronas) |neurona| {
            try std.testing.expectEqual(neurona.pesos.len, num_entradas);
            try std.testing.expectEqual(neurona.sesgo, 0);

            for (neurona.pesos) |peso| {
                try std.testing.expectApproxEqAbs(@as(f32, 0.5), peso, 0.0001);
            }
        }
    }
}
