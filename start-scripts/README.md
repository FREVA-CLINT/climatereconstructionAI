# Scheduling on Mistral and Levante

In order to run the program on a remote computer, it should be executed in a container that provides all necessary python packages. For this project, a Singularity container was used. A documentation on setting up singularity, see https://sylabs.io/guides/3.0/user-guide/installation.html.

The singularity container needs to be build on a local system since it requires root privileges. Singularity containers have two different operation modes: Production and sandbox. If additional software needs to be added to the container, switch from production to sandbox mode, if the container should be deployed on an external system, switch to production mode.

- production to sandbox: `sudo singularity build --sandbox development production.sif`
- sandbox to production: `sudo singularity build production.sif development`

## Mistral
The file `mistral.yaml` contains all necessary conda packages to run the program on the Mistral system.
Start the training or testing process on Mistral, execute following from the project's root directory:

`sbatch start-*-mistral.sh`

## Levante
The file `levante.yaml` contains all necessary conda packages to run the program on the Mistral system.
Start the training or testing process on Levante, execute following from the project's root directory:

`sbatch start-*-levante.sh`
