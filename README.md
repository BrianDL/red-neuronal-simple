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

Por otro lado, esta implementación es ligeramente más compleja en comparación con enfoques más simples. Requiere un manejo cuidadoso de la memoria, lo que implica una mayor responsabilidad por parte del programador para gestionar correctamente la asignación y liberación de recursos.

```zig
const Neurona = struct {
    pesos: []f32,
    sesgo: f32,
    funcion_activacion: fn (f32) f32,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, num_entradas: usize) !Neurona {
        var pesos = try allocator.alloc(f32, num_entradas);
        return Neurona{
            .pesos = pesos,
            .sesgo = 0,
            .funcion_activacion = activation_functions.sigmoid,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Neurona) void {
        self.allocator.free(self.pesos);
    }
};

```


