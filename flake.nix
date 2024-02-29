{
  description = "neural networks lab";

  outputs = { self, nixpkgs, }:
    let system = "x86_64-linux";
    in {
      devShells.${system}.default = with import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
        let
          custom-python = python310.withPackages (p:
            with p; [
              ipykernel
              numpy
              torch
              torchvision
              wandb
              black
              pycuda
            ]);
        in mkShell {
          packages = [
            custom-python
            cudaPackages.cuda_nvcc

            cudaPackages.cuda_cccl
            cudaPackages.cudatoolkit
          ];
        };
    };
}
