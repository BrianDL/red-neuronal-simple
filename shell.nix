{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    zig
    # git
    # texlive.combined.scheme-full
    # gnumake
  ];

  shellHook = ''
    echo "Welcome to the red-neuronal-simple development environment!"
    echo "Zig version: $(zig version)"
    # echo "Git version: $(git --version)"
    # echo "LaTeX version: $(latex --version | head -n 1)"
    # echo "GNU Make version: $(make --version | head -n 1)"
  '';
}