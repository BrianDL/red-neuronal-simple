# red-neuronal-simple
Proyecto de sistemas dinámicos implementando una red neuronal simple.

### Objetivos del proyecto
El objetivo de este proyecto es implementar la red neuronal más simple posible en zig desde cero y crecer el proyecto lo suficiente como para aprender una función lineal o quizá algo más complejo.
Comenzamos por describir las características más básicas de una neurona.

### Características básicas de una neurona

Una neurona artificial es la unidad fundamental de una red neuronal. Para implementar una neurona desde cero, necesitamos entender y modelar las siguientes características y propiedades básicas:

1. Entradas (inputs): Una neurona recibe múltiples señales de entrada. Cada entrada representa un dato o característica del problema que estamos tratando de resolver.

2. Pesos (weights): Cada entrada está asociada con un peso. Los pesos son valores numéricos que determinan la importancia relativa de cada entrada.

3. Función de suma (summation function): La neurona suma todas las entradas multiplicadas por sus respectivos pesos. Esta operación se conoce como suma ponderada.

4. Sesgo (bias): Es un término adicional que se suma al resultado de la función de suma. El sesgo permite ajustar el umbral de activación de la neurona.

5. Función de activación (activation function): Esta función toma el resultado de la suma ponderada más el sesgo y produce la salida de la neurona. Algunas funciones de activación comunes son la función sigmoide, la función ReLU (Rectified Linear Unit) y la función tangente hiperbólica.

6. Salida (output): Es el resultado final producido por la neurona después de aplicar la función de activación.

Para implementar una red neuronal simple, necesitaremos:

1. Crear una estructura de datos para representar una neurona con sus propiedades (entradas, pesos, sesgo).
2. Implementar la función de suma ponderada.
3. Implementar una o más funciones de activación.
4. Crear una estructura para representar una capa de neuronas.
5. Implementar el proceso de propagación hacia adelante (forward propagation) para calcular la salida de la red.

En las siguientes secciones, describiremos cómo implementar estos componentes en Zig y cómo combinarlos para crear una red neuronal simple capaz de aprender una función lineal.

### La estructura de datos
Comenzamos por describir la estructura de datos a utilizar para representar una neurona y cómo podemos utilizar las ventajas de Zig para representar muchas neuronas que conformen una capa.

#### Estructura basada en slices con allocator

Esta estructura ofrece una mayor flexibilidad y control sobre la memoria, lo que la hace más eficiente para redes grandes. Además, facilita la creación dinámica de redes, permitiendo ajustar el tamaño y la configuración de la red según sea necesario durante la ejecución del programa.

Por otro lado, esta implementación es ligeramente más compleja en comparación con otros enfoques. Requiere un manejo cuidadoso de la memoria, lo que implica una mayor responsabilidad por parte del programador para gestionar correctamente la asignación y liberación de recursos.

```zig
const Neurona = struct {
    pesos: []f32,
    sesgo: f32,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, num_entradas: usize) !Neurona {
        var pesos = try allocator.alloc(f32, num_entradas);
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

```

### Función de Suma ponderada

La función de suma ponderada combina las entradas con sus respectivos pesos y añade el sesgo para producir la entrada neta de la neurona. Matemáticamente, se expresa como:

z = (x₁ * w₁) + (x₂ * w₂) + ... + (xₙ * wₙ) + b

Donde:
- z es la suma ponderada
- xᵢ son las entradas
- wᵢ son los pesos correspondientes
- b es el sesgo

En Zig, podemos implementar esta función de manera independiente, tomando la neurona como parámetro:

```zig
fn suma_ponderada(neurona: *const Neurona, entradas: []const f32) f32 {
    if (entradas.len != neurona.pesos.len) {
        @panic("El número de entradas no coincide con el número de pesos");
    }

    var suma: f32 = neurona.sesgo;
    for (neurona.pesos) |peso, i| {
        suma += peso * entradas[i];
    }
    return suma;
}
```

### Función de Activación: Sigmoide

Después de calcular la suma ponderada, la neurona aplica una función de activación para producir su salida. Una de las funciones de activación más comunes es la función sigmoide.

La función sigmoide tiene las siguientes características:

1. Es una función continua y diferenciable, lo que la hace adecuada para el aprendizaje basado en gradientes.
2. Mapea cualquier número real de entrada a un valor entre 0 y 1.
3. Tiene una forma de "S" que introduce no-linealidad en el modelo.

Matemáticamente, la función sigmoide se expresa como:

σ(x) = 1 / (1 + e^(-x))

Donde:
- σ(x) es el valor de salida de la función sigmoide
- x es el valor de entrada (en nuestro caso, la suma ponderada)
- e es la base del logaritmo natural (aproximadamente 2.71828)

Características clave de la función sigmoide:

- Para valores de entrada muy negativos, la salida se acerca a 0.
- Para valores de entrada muy positivos, la salida se acerca a 1.
- Alrededor de x = 0, la función tiene su pendiente más pronunciada.

La función sigmoide es particularmente útil cuando queremos interpretar la salida de la neurona como una probabilidad o cuando necesitamos valores de salida acotados entre 0 y 1.

```zig
const std = @import("std");
const math = std.math;

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + math.exp(-x));
}
```

### Estructura de datos para una capa

Una capa en nuestra red neuronal es esencialmente una colección de neuronas que operan en paralelo. Cada neurona en una capa recibe las mismas entradas pero procesa esas entradas de manera independiente basándose en sus propios pesos y sesgos.

Para representar una capa, podemos utilizar un slice de neuronas. Esto nos permite tener un número variable de neuronas en cada capa y manejar la memoria de manera eficiente.

Aquí está la estructura básica para una capa:

```zig
const Capa = struct {
    neuronas: []Neurona,
    funcion_activacion: fn (f32) f32,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, num_neuronas: usize, num_entradas: usize, funcion_activacion: fn (f32) f32) !Capa {
        var neuronas = try allocator.alloc(Neurona, num_neuronas);
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
```

### Propagación hacia adelante (Forward Propagation)

La propagación hacia adelante es el proceso por el cual una red neuronal calcula su salida dado un conjunto de entradas. Este proceso implica pasar los datos de entrada a través de cada capa de la red, aplicando las operaciones de suma ponderada y función de activación en cada neurona.

Para nuestra red neuronal simple, el proceso de propagación hacia adelante se puede describir en los siguientes pasos:

1. Recibir las entradas de la red.
2. Para cada capa en la red:
   a. Para cada neurona en la capa:
      - Calcular la suma ponderada de las entradas y los pesos de la neurona.
      - Añadir el sesgo a la suma ponderada.
      - Aplicar la función de activación de la capa al resultado.
   b. Las salidas de esta capa se convierten en las entradas de la siguiente capa.
3. La salida de la última capa es la salida final de la red.

```zig
const RedNeuronal = struct {
    capas: []Capa,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, configuracion: []const usize, funciones_activacion: []const fn(f32) f32) !RedNeuronal {
        var capas = try allocator.alloc(Capa, configuracion.len - 1);
        errdefer allocator.free(capas);

        for (capas) |*capa, i| {
            capa.* = try Capa.init(allocator, configuracion[i + 1], configuracion[i], funciones_activacion[i]);
        }

        return RedNeuronal{
            .capas = capas,
            .allocator = allocator,
        };
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

        for (self.capas) |capa| {
            var salida_capa = try self.allocator.alloc(f32, capa.neuronas.len);
            defer self.allocator.free(salida_capa);

            for (capa.neuronas) |neurona, j| {
                const suma = suma_ponderada(&neurona, salida_actual);
                salida_capa[j] = capa.funcion_activacion(suma);
            }

            self.allocator.free(salida_actual);
            salida_actual = try self.allocator.dupe(f32, salida_capa);
        }

        return salida_actual;
    }
};
```

## Entrenamiento de la Red Neuronal

Para entrenar nuestra red neuronal, necesitamos implementar los siguientes componentes:

1. Función de pérdida (Loss function)
2. Retropropagación (Backpropagation)
3. Optimizador (Optimizer)
4. Bucle de entrenamiento (Training loop)

### Función de pérdida

La función de pérdida mide qué tan bien está funcionando nuestra red neuronal. Para una tarea de regresión simple, podemos usar el Error Cuadrático Medio (Mean Squared Error, MSE):

```zig
fn mse(predicciones: []const f32, objetivos: []const f32) f32 {
    if (predicciones.len != objetivos.len) {
        @panic("Las longitudes de predicciones y objetivos no coinciden");
    }
    var suma_errores: f32 = 0;
    for (predicciones) |pred, i| {
        const error = pred - objetivos[i];
        suma_errores += error * error;
    }
    return suma_errores / @intToFloat(f32, predicciones.len);
}
```

### Retropropagación
La retropropagación es el algoritmo que calcula los gradientes de la función de pérdida con respecto a los pesos y sesgos de la red. Esto nos permite saber cómo ajustar los parámetros para reducir el error. La implementación de la retropropagación es más compleja y requiere calcular las derivadas parciales en cada capa.




### Optimizador
El optimizador usa los gradientes calculados por la retropropagación para actualizar los pesos y sesgos de la red. Un optimizador simple es el Descenso de Gradiente Estocástico (SGD):

```zig
fn sgd(red: *RedNeuronal, tasa_aprendizaje: f32, gradientes: []const f32) void {
    var indice_grad: usize = 0;
    for (red.capas) |*capa| {
        for (capa.neuronas) |*neurona| {
            for (neurona.pesos) |*peso| {
                peso.* -= tasa_aprendizaje * gradientes[indice_grad];
                indice_grad += 1;
            }
            neurona.sesgo -= tasa_aprendizaje * gradientes[indice_grad];
            indice_grad += 1;
        }
    }
}
```

### Bucle de entrenamiento
El bucle de entrenamiento junta todos estos componentes. En cada iteración:
1.
Realiza la propagación hacia adelante
2.
Calcula la pérdida
3.
Realiza la retropropagación para obtener los gradientes
4.
Actualiza los pesos y sesgos usando el optimizador


### Enfoque Inicial: Entrenamiento Simplificado

Para comenzar, implementaremos un método de entrenamiento básico sin backpropagation compleja. Este enfoque nos permitirá entender los fundamentos del aprendizaje en redes neuronales.

```zig
fn entrenar_simple(red: *RedNeuronal, entradas: []const []const f32, objetivos: []const []const f32, epocas: usize, tasa_aprendizaje: f32) !void {
    for (0..epocas) |epoca| {
        var perdida_total: f32 = 0;
        for (entradas) |entrada, i| {
            const prediccion = try red.propagar_adelante(entrada);
            defer red.allocator.free(prediccion);

            const perdida = mse(prediccion, objetivos[i]);
            perdida_total += perdida;

            // Ajuste simple de pesos y sesgos
            for (red.capas) |*capa| {
                for (capa.neuronas) |*neurona| {
                    for (neurona.pesos) |*peso, j| {
                        peso.* += tasa_aprendizaje * (objetivos[i][0] - prediccion[0]) * entrada[j];
                    }
                    neurona.sesgo += tasa_aprendizaje * (objetivos[i][0] - prediccion[0]);
                }
            }
        }
        std.debug.print("Época {}: Pérdida promedio = {d:.6}\n", .{epoca + 1, perdida_total / @intToFloat(f32, entradas.len)});
    }
}
```
