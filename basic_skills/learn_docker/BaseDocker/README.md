# HELLO Docker!

This repository aims to provide a seamless experience for creating and running customized Docker images tailored for lab usage. Our goal is to ensure that each user can have their unique Docker image, mirroring their user rights and requirements. All custom images are based on a common base image, which is pre-built using the `Makefile.admin` and pushed to our lab's [Docker Hub repository](https://hub.docker.com/repository/docker/mmathislab/utils/tags?page=1&ordering=last_updated).

Please check out `Dockerfile.base` to explore the base environment. This image comes fully loaded with essential tools and libraries, including TensorFlow, PyTorch, Python, Jupyter, DataJoint support, and more.

## Getting Started
0. Clone BaseDocker current rep from git.
1. First, run the following command to see all available commands for building and running containers:
   ```bash
   make help
   ```
   
2. Feel free to use this Makefile to ensure consistent naming for your Docker images and containers. You can copy the Makefile and adjust it according to your specific needs.
3. Before building your customized image, open `Dockerfile.core` and add any additional packages you require. (For example, CEBRA).
   ```bash 
      RUN pip3 install cebra #add to Dockerfile.core
   ```
4. For DLC consider using the official DLC images from the [DLC Docker Hub repository](https://hub.docker.com/r/deeplabcut/deeplabcut/tags). You need to adjust the FROM clause in `Dockerfile.core` to inherit from the DLC image while maintaining your customized environment.
5. In `Dockerfile.core`, carefully review all [MODIFY/ADD] comments, make the necessary adjustments, and save the file.
6. Now, open the `Makefile` for editing. You should specify the following parameters:
-    `PROJECT_NAME`: Your project's name.
-    `Volumes` to share code and data with the Docker container.
-    Choose an appropriate `tag` for your image and containers.
- Do not modify any variables unless they are explicitly marked as [USER:...]!

7. To build your Docker image, run:
```bash
make build
```
8. To run a container based on your image, execute:
```bash
 make run
```
The default command to execute is bash, so now you are inside the container! Run `start_jupyter` if you need a jupyter notebook.

9. If you plan to use Jupyter Notebook, open a second terminal on your host machine and run the following command (replace ssh_host with the name of the host from your SSH config file):
    
```bash
ssh -NL 8888:localhost:$(PORT) ssh_host &
```
From server's point of view it selects available port automatically via `find_ports.sh` script.

10. ## Working with VSCode and Docker

If you prefer to use VSCode for your development workflow while working with Docker, you can follow these steps:

1. Refer to the official [VSCode documentation](https://code.visualstudio.com/docs/containers/overview) on how to attach a Dockerfile and create a configuration for your project.

2. When using VSCode with Docker, please be mindful of the following:

   - Ensure you close all VSCode sessions correctly after usage.

   - Verify that there are no lingering VSCode processes using the `ps -aux` command before exiting your development environment.

Using VSCode in conjunction with Docker can provide a seamless development experience. Be sure to consult the official documentation for detailed instructions on setting up your environment.
