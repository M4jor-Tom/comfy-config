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

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            (python312.withPackages (ps: with ps; [ pyyaml pytest ]))
            openssh
            jq
          ];
          shellHook = ''
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            echo "comfy-runpod dev shell — RUNPOD_API_KEY ''${RUNPOD_API_KEY:+is set}"
          '';
        };
      });
    };
}
