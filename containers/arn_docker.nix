# Build a Docker image using Nix and dockerTools.
{ pkgs ? import <nixpkgs> { } 
, pkgsLinux ? import <nixpkgs> { system = "x86_64-linux"; }
}:

let
/*
  peanut = pkgs.dockerTools.pullImage {
  imageName = "nvcr.io/nvidia/pytorch";
  imageDigest = "sha256:ab303284646e417327ae1589b3768e04211669438110efec6c7fccad476b4b18";
  sha256 = "053pf4y23aa821sj4mc5lkl6nxs9vg40rr0k7bq3snlyryw9igsl";
  finalImageName = "nvcr.io/nvidia/pytorch";
  finalImageTag = "21.09-py3";
};
  #python-with-pkgs = python3.withPackages python-pkgs;
  python_pkgs = with pkgs.python39Packages; [
    numpy
    scipy
    scikit-learn
    pandas
    h5py
    tqdm
    torchWithCuda
    torchvision
    #torchsummary
    torchmetrics
    pytorch-lightning
    pyro-ppl
    cython
    hdbscan
    opencv4
    pyaml
    #(opencv4.override { enableGtk2 = true; })
  ]; 
#*/
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {
    python = "python39";
  };
in
pkgs.dockerTools.buildImage {
  name = "arn";
  tag = "0.2.0rc1";

  contents = (builtins.toList mach-nix.mkNixpkgs {
    requirements = builtins.readFile ../requirements/arn.txt;
  });
  runAsRoot = ''
    #!${pkgs.runtimeShell}
    # Install Prijatelj's public fork of `vast` for the Extreme Value
    # Machine and FINCH with recurse-submodules, and get pyflann as dep.
    git clone https://github.com/primetang/pyflann.git

    # Have to 2to3 the pyflann code...
    pip install 2to3==1.0
    2to3 pyflann/
    pip install -e pyflann/

    git clone --recurse-submodules https://github.com/prijatelj/vast
    pip install -e vast/
  '';

  config = {
    #Cmd = [ "/bin/..." ];
    WorkingDir = "/arn";
    #Volumes = { "/arn" = { }; };
  };
}
