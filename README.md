# PhD-thesis

Repository that collects my Doctorate dissertation.

## Compilation
To ease the compilation we provide some ``Makefile`` rules, in particular:

- `make` compiles the current working copy;
- `make world` compiles the whole document, regenerating the bibliography at
  the same time.

## Git submodules
In order to compile the final document, first ensure yourself that Git
submodules are correctly fetched with the command
```shell
git submodule init && git submodule update
```
to be ran after the first checkout, thereafter they can be updated as usual.
After that, `make update-submodules` can be used to recursively update
submodules.

