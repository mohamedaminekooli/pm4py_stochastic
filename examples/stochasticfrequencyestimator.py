import pm4py
import os
#from pm4py.objects.petri_net.stochastic.weightestimators import abstractfrequencyestimator
import pm4py.objects.petri_net.stochastic.common.visualize as visualize
from pm4py.estimators import discover_stochastic_petrinet

# New function to discover the stochastic process
#def discover_stochastic_petrinet(log, pn):
 #   discoverer = abstractfrequencyestimator.AbstractFrequencyEstimator()
  #  return discoverer.estimate_weights_apply(log, pn)
    
def use_inductive_miner_petrinet_discovery(log):
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=0.2)
    pm4py.view_process_tree(tree, format="svg")
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return net, im, fm
    
def execute_script():
    log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "running-example.xes"))
    net, im, fm = use_inductive_miner_petrinet_discovery(log)
    spn = discover_stochastic_petrinet(log,net)
    visualize.apply(spn, im, fm, format="svg")
    #pm4py.view_petri_net(spn, im, fm, format="svg")


if __name__ == "__main__":
    execute_script()