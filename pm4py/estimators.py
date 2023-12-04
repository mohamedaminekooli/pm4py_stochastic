from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace, Event, EventStream
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py.objects.petri_net.stochastic.weightestimators import abstractfrequencyestimator
import pm4py.objects.petri_net.stochastic.common.visualize as visualize 


def discover_stochastic_petrinet(log: Union[EventLog, pd.DataFrame], petri_net: PetriNet, initial_marking: Marking,
                                               final_marking: Marking, activity_key: str = "concept:name", timestamp_key: str = "time:timestamp", case_id_key: str = "case:concept:name") -> Tuple[StochasticPetriNet, Marking, Marking]:
    """
    Discover a stochastic Petri net using weight estimators and a petri net discovered using Inductive Miner

    :param log: event log / Pandas dataframe
    :param petri_net: resulting petri net from inductive miner 
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``Tuple[StochasticPetriNet, Marking, Marking]``

    .. code-block:: python3

        import pm4py

        net, im, fm = pm4py.discover_stochastic_petrinet(dataframe, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception(
            "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    spn_parameters = abstractfrequencyestimator.Parameters
    parameters = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)
    discoverer = abstractfrequencyestimator.AbstractFrequencyEstimator()
    return discoverer.estimate_weights_apply(log, petri_net, parameters)