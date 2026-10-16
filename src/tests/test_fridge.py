import warnings

import pytest
from networkx.classes.graph import Graph

from railgen.constants import GRAPH_FILE, RAIL_NETWORK_DIR, YARD_GROUPS_DICT_FILE
from railgen.models import ygs_dict_adapter
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.schedule_config import Config

NO_NETWORK = not all(
    (RAIL_NETWORK_DIR / f).exists() for f in (GRAPH_FILE, YARD_GROUPS_DICT_FILE)
)
NO_NETWORK_MSG = f"No built rail network in {RAIL_NETWORK_DIR}/, run `fridge build-network` to test it"
if NO_NETWORK:
    warnings.warn(NO_NETWORK_MSG)


@pytest.mark.skipif(NO_NETWORK, reason=NO_NETWORK_MSG)
def test_building_graph_and_ygs_dict():
    network = RailNetwork(Config())

    # Verify that the graph is a Graph object
    assert ygs_dict_adapter.validate_python(network.ygs_dict)
    assert isinstance(network.ygs_graph, Graph)

    # Verify correct number of nodes
    assert len(network.ygs_graph.nodes) == len(network.ygs_dict)
