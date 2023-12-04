from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pm4py
from enum import Enum
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.statistics.start_activities.pandas import get as start_activities_get
from pm4py.statistics.end_activities.pandas import get as end_activities_get
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.conversion.log import converter


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

class FrequencyCalculator:
    def __init__(self, log: EventLog, parameters: Optional[Dict[Any, Any]] = None):
        if parameters is None:
            parameters = {}
        self.log = log
        self.parameters = parameters

    def scan_log(self, spn: StochasticPetriNet):
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(self.log, activity_key, parameters=self.parameters)
        for transition in spn.transitions:
            if transition.label not in activities_occurrences:
                activities_occurrences[transition.label] = 1
        return activities_occurrences

    def calculate_follows_frequency(self, current_activity, end_activities):
        follows_frequency = {}
        eventlog = converter.apply(self.log, variant=converter.Variants.TO_EVENT_LOG, parameters=self.parameters)
        for trace in eventlog:
            activities = [event["concept:name"] for event in trace]
            for i in range(len(activities) - 1):
                if activities[i] == current_activity and activities[i + 1] in end_activities:
                    follows_frequency[current_activity] = follows_frequency.get(current_activity, 0) + 1
        return follows_frequency

    def calculate_start_frequency(self, current_activity):
        start_frequency = {}
        start_activities = start_activities_get.get_start_activities(self.log, parameters=self.parameters)
        if current_activity in start_activities:
            start_frequency[current_activity] = start_activities[current_activity]
        else:
            start_frequency[current_activity] = 0
        return start_frequency

    def calculate_end_frequency(self, current_activity):
        end_frequency = {}
        end_activities = end_activities_get.get_end_activities(self.log, parameters=self.parameters)
        if current_activity in end_activities:
            end_frequency[current_activity] = end_activities[current_activity]
        else:
            end_frequency[current_activity] = 0
        return end_frequency

    def calculate_wrhpair(self):
        activities_weights = {}
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(log, activity_key, parameters=self.parameters)
        end_activities = end_activities_get.get_end_activities(self.log, parameters=self.parameters)
        activities = list(activities_occurrences.keys())
        for current_activity in activities:
            start_frequency = self.calculate_start_frequency(current_activity)
            end_frequency = self.calculate_end_frequency(current_activity)
            follows_frequency = self.calculate_follows_frequency(current_activity,end_activities)
            total_follows_frequency = follows_frequency[current_activity] if current_activity in follows_frequency else 0
            weight = max(1, start_frequency[current_activity] + end_frequency[current_activity] + total_follows_frequency)
            activities_weights[current_activity] = weight
        return activities_weights

class ActivityPairRHWeightEstimator:
    def __init__(self):
        self.activities_weights = {}

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        frequency_calculator = FrequencyCalculator(log)
        self.activities_weights = frequency_calculator.calculate_wrhpair()
        spn = StochasticPetriNet(pn)
        spn = pn
        return self.estimate_activity_pair_weights(spn)

    # Assign weights to transitions based on event frequencies
    def estimate_activity_pair_weights(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activities_weights(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activities_weights(self, tran):
        activity = tran.label
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = float(self.activities_weights.get(activity, 1))
        return frequency


from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace, Event, EventStream
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
import os
def discover_stochastic_petrinet(log: Union[EventLog, pd.DataFrame], petri_net: PetriNet, initial_marking: Marking,
                                               final_marking: Marking, activity_key: str = "concept:name", timestamp_key: str = "time:timestamp", case_id_key: str = "case:concept:name") -> Tuple[StochasticPetriNet, Marking, Marking]:
    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception(
            "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    spn_parameters = Parameters
    parameters = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)
        discoverer = ActivityPairRHWeightEstimator()
        return discoverer.estimate_weights_apply(log, petri_net)
    
def use_inductive_miner_petrinet_discovery(log):
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=0.2)
    pm4py.view_process_tree(tree, format="svg")
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return net, im, fm

#----------------------------------------------------------------------------------------------------------

import tempfile

from graphviz import Digraph

from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.util import exec_utils, constants
from enum import Enum
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_TIMESTAMP_KEY, DEFAULT_ARTIFICIAL_START_ACTIVITY, DEFAULT_ARTIFICIAL_END_ACTIVITY


class Parameters_view(Enum):
    FORMAT = "format"
    DEBUG = "debug"
    RANKDIR = "set_rankdir"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = PARAMETER_CONSTANT_TIMESTAMP_KEY
    AGGREGATION_MEASURE = "aggregationMeasure"
    FONT_SIZE = "font_size"
    BGCOLOR = "bgcolor"
    DECORATIONS = "decorations"


def apply_algo(net:StochasticPetriNet, initial_marking, final_marking, decorations=None, parameters=None):
    """
    Apply method for Petri net visualization (it calls the
    graphviz_visualization method)

    Parameters
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    decorations
        Decorations for elements in the Petri net
    parameters
        Algorithm parameters

    Returns
    -----------
    viz
        Graph object
    """
    if parameters is None:
        parameters = {}

    image_format = exec_utils.get_param_value(Parameters_view.FORMAT, parameters, "png")
    debug = exec_utils.get_param_value(Parameters_view.DEBUG, parameters, False)
    set_rankdir = exec_utils.get_param_value(Parameters_view.RANKDIR, parameters, None)
    font_size = exec_utils.get_param_value(Parameters_view.FONT_SIZE, parameters, "12")
    bgcolor = exec_utils.get_param_value(Parameters_view.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR)

    if decorations is None:
        decorations = exec_utils.get_param_value(Parameters_view.DECORATIONS, parameters, None)
    return graphviz_visualization(net, image_format=image_format, initial_marking=initial_marking,
                                  final_marking=final_marking, decorations=decorations, debug=debug,
                                  set_rankdir=set_rankdir, font_size=font_size, bgcolor=bgcolor)


def graphviz_visualization(net:StochasticPetriNet, image_format="png", initial_marking=None, final_marking=None, decorations=None,
                           debug=False, set_rankdir=None, font_size="12", bgcolor=constants.DEFAULT_BGCOLOR):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if decorations is None:
        decorations = {}

    font_size = str(font_size)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    viz = Digraph(net.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})
    if set_rankdir:
        viz.graph_attr['rankdir'] = set_rankdir
    else:
        viz.graph_attr['rankdir'] = 'LR'

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        label = decorations[t]["label"] if t in decorations and "label" in decorations[t] else ""
        fillcolor = decorations[t]["color"] if t in decorations and "color" in decorations[t] else None
        textcolor = "black"

        if t.label is not None and not label:
            label = t.label
        if debug:
            label = t.name
        label = str(label)

        if fillcolor is None:
            if t.label is None:
                fillcolor = "black"
                if label:
                    textcolor = "white"
            else:
                fillcolor = bgcolor

        # Add transition weight to the label
        weight = decorations[t]["weight"] if t in decorations and "weight" in decorations[t] else 1
        if t.weight is not None:
            weight = t.weight
        if debug:
            weight = t.weight
        weight = str(weight)
        if t.label is not None:
            label += f" ({weight})"
        else:
            # Create a separate node for the weight and connect it to the transition
            weight_node_name = f"Weight_{t}"
            viz.node(weight_node_name, f"Weight: {weight}", shape='plaintext')
            viz.edge(weight_node_name, str(id(t)), style='dashed', constraint='false', len='0.1')

        viz.node(str(id(t)), label, style='filled', fillcolor=fillcolor, border='1', fontsize=font_size, fontcolor=textcolor)

        if petri_properties.TRANS_GUARD in t.properties:
            guard = t.properties[petri_properties.TRANS_GUARD]
            viz.node(str(id(t))+"guard", style="dotted", label=guard)
            viz.edge(str(id(t))+"guard", str(id(t)), arrowhead="none", style="dotted")

    # places
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        label = decorations[p]["label"] if p in decorations and "label" in decorations[p] else ""
        fillcolor = decorations[p]["color"] if p in decorations and "color" in decorations[p] else bgcolor

        label = str(label)
        if p in initial_marking:
            if initial_marking[p] == 1:
                viz.node(str(id(p)), "<&#9679;>", fontsize="34", fixedsize='true', shape="circle", width='0.75', style="filled", fillcolor=fillcolor)
            else:
                viz.node(str(id(p)), str(initial_marking[p]), fontsize="34", fixedsize='true', shape="circle", width='0.75', style="filled", fillcolor=fillcolor)
        elif p in final_marking:
            # <&#9632;>
            viz.node(str(id(p)), "<&#9632;>", fontsize="32", shape='doublecircle', fixedsize='true', width='0.75', style="filled", fillcolor=fillcolor)
        else:
            if debug:
                viz.node(str(id(p)), str(p.name), fontsize=font_size, shape="ellipse")
            else:
                if p in decorations and "color" in decorations[p] and "label" in decorations[p]:
                    viz.node(str(id(p)), label, style='filled', fillcolor=fillcolor,
                             fontsize=font_size, shape="ellipse")
                else:
                    viz.node(str(id(p)), label, shape='circle', fixedsize='true', width='0.75', style="filled", fillcolor=fillcolor)

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))

    # check if there is an arc with weight different than 1.
    # in that case, all the arcs in the visualization should have the arc weight visible
    arc_weight_visible = False
    for arc in arcs_sort_list:
        if arc.weight != 1:
            arc_weight_visible = True
            break

    for a in arcs_sort_list:
        penwidth = decorations[a]["penwidth"] if a in decorations and "penwidth" in decorations[a] else None
        label = decorations[a]["label"] if a in decorations and "label" in decorations[a] else ""
        color = decorations[a]["color"] if a in decorations and "color" in decorations[a] else None

        if not label and arc_weight_visible:
            label = a.weight

        label = str(label)
        arrowhead = "normal"

        if petri_properties.ARCTYPE in a.properties:
            if a.properties[petri_properties.ARCTYPE] == petri_properties.RESET_ARC:
                arrowhead = "vee"
            elif a.properties[petri_properties.ARCTYPE] == petri_properties.INHIBITOR_ARC:
                arrowhead = "dot"

        viz.edge(str(id(a.source)), str(id(a.target)), label=label,
                 penwidth=penwidth, color=color, fontsize=font_size, arrowhead=arrowhead, fontcolor=color)

    viz.attr(overlap='false')

    viz.format = image_format.replace("html", "plain-ext")

    return viz


def view_stochastic_petri_net(petri_net: StochasticPetriNet, initial_marking: Optional[Marking] = None,
                   final_marking: Optional[Marking] = None, format: str = constants.DEFAULT_FORMAT_GVIZ_VIEW, bgcolor: str = "white",
                   decorations: Dict[Any, Any] = None, debug: bool = False, rankdir: str = constants.DEFAULT_RANKDIR_GVIZ):
    format = str(format).lower()
    #from pm4py.visualization.petri_net import visualizer as pn_visualizer
    gviz = apply(petri_net, initial_marking, final_marking,
                               parameters={Variants.WO_DECORATION.value.Parameters.FORMAT: format, "bgcolor": bgcolor, "decorations": decorations, "debug": debug, "set_rankdir": rankdir})
    view(gviz)
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave
from pm4py.visualization.petri_net.variants import wo_decoration, alignments, greedy_decoration_performance, \
    greedy_decoration_frequency, token_decoration_performance, token_decoration_frequency
from pm4py.util import exec_utils
from enum import Enum
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog, EventStream
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.visualization.common.gview import serialize, serialize_dot
import graphviz
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet


class Variants(Enum):
    WO_DECORATION = wo_decoration
    FREQUENCY = token_decoration_frequency
    PERFORMANCE = token_decoration_performance
    FREQUENCY_GREEDY = greedy_decoration_frequency
    PERFORMANCE_GREEDY = greedy_decoration_performance
    ALIGNMENTS = alignments


WO_DECORATION = Variants.WO_DECORATION
FREQUENCY_DECORATION = Variants.FREQUENCY
PERFORMANCE_DECORATION = Variants.PERFORMANCE
FREQUENCY_GREEDY = Variants.FREQUENCY_GREEDY
PERFORMANCE_GREEDY = Variants.PERFORMANCE_GREEDY
ALIGNMENTS = Variants.ALIGNMENTS


def apply(net: StochasticPetriNet, initial_marking: Marking = None, final_marking: Marking = None, log: Union[EventLog, EventStream, pd.DataFrame] = None, aggregated_statistics=None, parameters: Optional[Dict[Any, Any]] = None,
          variant=Variants.WO_DECORATION) -> graphviz.Digraph:
    if parameters is None:
        parameters = {}
    if log is not None:
        if isinstance(log, pd.DataFrame):
            log = dataframe_utils.convert_timestamp_columns_in_df(log)

        log = log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG)
    return apply_viz(net, initial_marking, final_marking, log=log,
                                                 aggregated_statistics=aggregated_statistics,
                                                 parameters=parameters)

def apply_viz(net: PetriNet, initial_marking: Marking, final_marking: Marking, log: EventLog = None, aggregated_statistics=None, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> graphviz.Digraph:
    """
    Apply method for Petri net visualization (it calls the graphviz_visualization
    method) adding performance representation obtained by token replay

    Parameters
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    log
        (Optional) log
    aggregated_statistics
        Dictionary containing the frequency statistics
    parameters
        Algorithm parameters (including the activity key used during the replay, and the timestamp key)

    Returns
    -----------
    viz
        Graph object
    """
    if aggregated_statistics is None:
        if log is not None:
            aggregated_statistics = token_decoration_frequency.get_decorations(log, net, initial_marking, final_marking, parameters=parameters,
                                                    measure="performance")
    return apply_algo(net, initial_marking, final_marking, parameters=parameters,
                           decorations=aggregated_statistics)

def save(gviz: graphviz.Digraph, output_file_path: str, parameters=None):
    """
    Save the diagram

    Parameters
    -----------
    gviz
        GraphViz diagram
    output_file_path
        Path where the GraphViz output should be saved
    """
    gsave.save(gviz, output_file_path, parameters=parameters)


def view(gviz: graphviz.Digraph, parameters=None):
    """
    View the diagram

    Parameters
    -----------
    gviz
        GraphViz diagram
    """
    return gview.view(gviz, parameters=parameters)

# Usage example:
log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "example_12.xes"))
net, im, fm = use_inductive_miner_petrinet_discovery(log)
spn = discover_stochastic_petrinet(log, net, im, fm)
view_stochastic_petri_net(spn, im, fm, format="svg")
