{
  description = "comfy-runpod — ComfyUI generation harness on Runpod";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      packages = forAll (pkgs: {
        default = pkgs.python312Packages.buildPythonApplication {
          pname = "comfy-runpod";
          version = "0.1.0";
          src = ./.;
          pyproject = true;
          build-system = [ pkgs.python312Packages.setuptools ];
          dependencies = [ pkgs.python312Packages.pyyaml ];
          # ponytail: ssh is called via subprocess, so it must be on PATH at runtime
          makeWrapperArgs = [ "--prefix PATH : ${pkgs.openssh}/bin" ];
          doCheck = false;
          # pname (comfy-runpod) differs from the console script (comfy); tell
          # `nix run` which binary to execute instead of guessing from pname.
          meta.mainProgram = "comfy";
        };
      });

      devShells = forAll (pkgs:
        let
          pythonEnv = pkgs.python312.withPackages (ps: with ps; [ pyyaml pytest ]);
          # A thin shim, not `packages.default`: the built package would bake in
          # a snapshot of src/ at build time, shadowing the PYTHONPATH trick
          # below and defeating live-editing -- the entire point of a dev shell.
          comfyBin = pkgs.writeShellScriptBin "comfy" ''
            exec ${pythonEnv}/bin/python -m comfy_runpod.cli "$@"
          '';
        in {
          default = pkgs.mkShell {
            packages = [ pythonEnv comfyBin pkgs.openssh pkgs.jq ];
            shellHook = ''
              export PYTHONPATH=$PWD/src:$PYTHONPATH
              echo "comfy-runpod dev shell — RUNPOD_API_KEY ''${RUNPOD_API_KEY:+is set}"
            '';
          };
        });
    };
}
