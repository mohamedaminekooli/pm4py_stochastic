from pm4py.objects.petri_net.stochastic.weightestimators.abstractfrequencyestimator import view_stochastic_petri_net
from pm4py.objects.petri_net.stochastic.weightestimators.abstractfrequencyestimator import use_inductive_miner_petrinet_discovery
from pm4py.objects.petri_net.stochastic.weightestimators.abstractfrequencyestimator import discover_stochastic_petrinet
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet

def export_petri_to_slpn(StochasticPetriNet, marking, output_filename):
    """
    Export a StochasticPetriNet to an SLPN file

    Parameters
    ----------
    StochasticPetriNet: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    marking: :class:`pm4py.entities.petri.petrinet.Marking`
        Marking
    output_filename:
        Absolute output file name for saving the slpn file
    """
    num_places = len(StochasticPetriNet.places)
    num_transitions = len(StochasticPetriNet.transitions)

    with open(output_filename, "w") as file:
        # Write the number of places
        file.write(f"{num_places}\n")

        # Write the initial marking
        for place in StochasticPetriNet.places:
            marking_value = marking[place] if place in marking else 0
            file.write(f"{marking_value}\n")

        # Write the number of transitions
        file.write(f"{num_transitions}\n")

        # Write information for each transition
        for transition in StochasticPetriNet.transitions:
            file.write(f"# transition {transition.name}\n")

            # Write the label (silent or actual label)
            if transition.label is not None:
                file.write(f"label {transition.label}\n")
            else:
                file.write("silent\n")

            # Write the weight
            file.write(f"# weight\n{transition.weight}\n")

            # Write the number of input places
            file.write(f"# number of input places\n{len(transition.in_arcs)}\n")

            # Write the input places
            for arc in transition.in_arcs:
                file.write(f"{StochasticPetriNet.places.index(arc.source)}\n")

            # Write the number of output places
            file.write(f"# number of output places\n{len(transition.out_arcs)}\n")

            # Write the output places
            for arc in transition.out_arcs:
                file.write(f"{StochasticPetriNet.places.index(arc.target)}\n")

# Example usage
# Assuming you have a Petri net 'petri_net' and its initial marking 'initial_marking'
import pm4py
import os
log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "example_12.xes"))
net, im, fm = use_inductive_miner_petrinet_discovery(log)
spn = discover_stochastic_petrinet(log, net, im, fm)
view_stochastic_petri_net(net, im, fm, format="svg")
export_petri_to_slpn(spn, im, "output.slpn")
