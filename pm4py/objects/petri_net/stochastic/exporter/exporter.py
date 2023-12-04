'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
from enum import Enum

from pm4py.objects.petri_net.stochastic.exporter.variants import slpn
from pm4py.util import exec_utils


class Variants(Enum):
    SLPN = slpn


SLPN = Variants.SLPN


def apply(net, initial_marking, output_filename, final_marking=None, variant=SLPN, parameters=None):
    """
    Export a Stochastic Petri Net along with an initial marking (and possibly a final marking) to an output file

    Parameters
    ------------
    net
        Stochastic Petri Net
    initial_marking
        Initial marking
    output_filename
        Output filename
    final_marking
        Final marking
    variant
        Variant of the algorithm, possible values:
            - Variants.SLPN
    parameters
        Parameters of the exporter
    """
    return exec_utils.get_variant(variant).export_petri_to_slpn(net, initial_marking, output_filename,
                                                      final_marking=final_marking, parameters=parameters)