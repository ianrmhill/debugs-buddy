# debugs-buddy
Guided analog circuit debugger using Bayesian optimal experiment design

## Code Overview
Currently the repository is in a state of flux. The guided debugger is in the 'debugsbuddy' source folder, with file separation into the inference+BOED part in 'core', the circuit component models in 'components', and the linear circuit solver in 'circuits'. Example guided debug flows for specific circuit topologies are in the 'demos' folder, and then the 'boed\_dev\_examples' folder contains interesting smaller scale problems that were tackled in preparation for the larger debugs-buddy project.

## Getting Started
The likely best way to get familiar with the project is to run the three-res-demo example circuit debug to understand the user flow and see how the full program is structured.

## Adding New Circuit Topologies for Debug
Please read the following topics prior to implementing your own circuits for debugging

### Quantities that Influence Debugging Quality
The behaviour/performance of the debugger is dependent on many parameters, correct tuning of the following is crucial for obtaining fast and accurate debugging performance:
* Problem specific:
  - Total number of circuit nodes (increasing number of latent variables makes inferring specific faults much more challenging, debugger will require more time, more iterations, and will be less certain about its conclusions)
  - Number of observation nodes (more useful observations massively increases performance of the debugger)
  - Parameter magnitudes (large or small values for component parameters can quickly lead to instability, try to normalize to around 1 as much as possible)
  - Total number of input sources (more sources increases the experiment design space rapidly and thus slows down the debugger)
  - Number of faults and rate of 'equivalent' faults in circuit topology (more faults or many possible faults that result in identical behaviour make identification of the specific fault more challenging)
* Internal method quantities:
  - Sample quantities for both 'layers' of EIG calculation and inference (more samples increases estimate accuracy for both but also slows down the debugger and may result in memory limitation issues)
  - Change in belief thresholds for user reporting (too high and debugger won't provide rapid feedback, too low and reported possible faults may be inaccurate)
  - Level of discretization of input experiment design space (an input voltage from 0 to 1V can be broken into 100mV steps or 10mV steps, more steps means better identification of the best experiment but slows down the debugger)
  - Initial prior beliefs for open/short fault probability and possible parameter values (very nuanced discussion, affects chances of sampling the correct faulty circuit, bias towards blaming certain fault types, and rate of inference)
  - Short and open resistances in fault model (to avoid singular matrices shorts and opens must have non-zero and finite resistances respectively; too close to one and the circuit solver becomes inaccurate, too far and the solver becomes unstable)
  - Measurement uncertainty of observed quantities (increased instrument error / uncertainty improves stability of debugger but reduces speed of inference thus requiring more iterations)
        
