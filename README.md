Este proyecto implementa una red neuronal desde cero en zig.
Para ver el resultado en acción sólo es necesario clonar este repositorio:

```
git clone https://github.com/BrianDL/red-neuronal-simple
cd red-neuronal-simple
```

Luego sólo se necesita tener zig instalado. Si no lo tiene, siempre se puede utilizar nix para instalarlo, si es que ya se tiene nix en el sistema:

```
nix-shell
```

Por último, se puede editar el archivo src/main.zig para cambiar los inputs de entrenamiento de la red neuronal y luego correr ```zig build run``` para verla en acción.