# Deployment Instructions
These are the docker-compose-based deployment instructions for this Super Kit. There are two different types of deployment, slim and full: 
- Deployment: Prepares data and launches services. 
  1. Navigate to the **/setup** directory
  2. Run `make install-tools` to setup dependencies
  3. Run `make launch` to launch the application
  4. Front-end is hard-coded to run on port 5005 - open http://localhost:5005 on your local browser to access the frontend.
  5. To shutdown app, run **make shutdown** - this will shutdown and remove docker containers but leaves images.