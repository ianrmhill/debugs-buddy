# debugs-buddy
Guided analog circuit debugger using Bayesian optimal experiment design

## Code Overview
Currently the repository is in a state of flux. The guided debugger is in the 'debugsbuddy' source folder, with file separation into the inference+BOED part in 'core', the circuit component models in 'components', and the linear circuit solver in 'circuits'. Example guided debug flows for specific circuit topologies are in the 'demos' folder, and then the 'boed\_dev\_examples' folder contains interesting smaller scale problems that were tackled in preparation for the larger debugs-buddy project.

## Getting Started
The likely best way to get familiar with the project is to run the three-res-demo example circuit debug to understand the user flow and see how the full program is structured.
