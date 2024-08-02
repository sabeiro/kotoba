import logging
import os
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Any
from falkordb import FalkorDB as FalkorDBDriver
from neo4j import GraphDatabase

class Driver(ABC):
    @abstractmethod
    def get_graph_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Abstract method to get graph data.
        """
        pass

    @abstractmethod
    def get_graph_history(self, skip: int, per_page: int) -> dict[str, Any]:
        """
        Abstract method to get graph history.

        :param skip: The number of items to skip.
        :param per_page: The number of items per page.
        :return: The graph history data.
        """
        pass

    @abstractmethod
    def get_response_data(
            self, response_data: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Abstract method to process response data.

        :param response_data: The response data to process.
        :return: The processed response data.
        """
        pass

class Metadata(BaseModel):
    createdDate: str = Field(
        ..., description="The date the knowledge graph was created"
    )
    lastUpdated: str = Field(
        ..., description="The date the knowledge graph was last updated"
    )
    description: str = Field(..., description="Description of the knowledge graph")


class Node(BaseModel):
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Label for the node")
    type: str = Field(..., description="Type of the node")
    color: str = Field(..., description="Color for the node")
    properties: Dict[str, Any] = Field(
        {}, description="Additional attributes for the node"
    )


class Edge(BaseModel):
    # WARING: Notice that this is "from_", not "from"
    from_: str = Field(..., alias="from", description="Origin node ID")
    to: str = Field(..., description="Destination node ID")
    relationship: str = Field(..., description="Type of relationship between the nodes")
    direction: str = Field(..., description="Direction of the relationship")
    color: str = Field(..., description="Color for the edge")
    properties: Dict[str, Any] = Field(
        {}, description="Additional attributes for the edge"
    )


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships.
    Use the colors to help differentiate between different node or edge types/categories.
    Always provide light pastel colors that work well with black font.
    """

    metadata: Metadata = Field(..., description="Metadata for the knowledge graph")
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    edges: List[Edge] = Field(..., description="List of edges in the knowledge graph")


class FalkorDB(Driver):
    def __init__(self):
        url = os.environ.get("FALKORDB_URL", "redis://localhost:6379")

        self.driver = FalkorDBDriver.from_url(url).select_graph("falkordb")

        # Check if connection is successful
        try:
            logging.info("FalkorDB database connected successfully!")
        except ValueError as ve:
            logging.error("FalkorDB database: {}".format(ve))
            raise

    def get_graph_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes = self.driver.query(
            """
        MATCH (n)
        RETURN {data: {id: n.id, label: n.label, color: n.color}}
        """
        )
        nodes = [el[0] for el in nodes.result_set]

        edges = self.driver.query(
            """
        MATCH (s)-[r]->(t)
        return {data: {source: s.id, target: t.id, label:r.type, color: r.color}}
        """
        )
        edges = [el[0] for el in edges.result_set]

        return (nodes, edges)

    def get_graph_history(self, skip, per_page) -> dict[str, Any]:
        # Getting the total number of graphs
        result = self.driver.query(
            """
        MATCH (n)-[r]->(m)
        RETURN count(n) as total_count
        """
        )

        total_count = result.result_set[0][0]

        # If there is no history, return an empty list
        if total_count == 0:
            return {"graph_history": [], "remaining": 0, "graph": True}

        # Fetching 10 most recent graphs
        result = self.driver.query(
            """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        ORDER BY r.timestamp DESC
        SKIP {skip}
        LIMIT {per_page}
        """.format(
            skip=skip, per_page=per_page
        )
        )

        # Process the 'result' to format it as a list of graphs
        graph_history = [
            FalkorDB._process_graph_data(record) for record in result.result_set
        ]
        remaining = max(0, total_count - skip - per_page)

        return {"graph_history": graph_history, "remaining": remaining, "graph": True}

    def get_response_data(
            self, response_data
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Import nodes
        nodes = self.driver.query(
            """
            UNWIND $nodes AS node
            MERGE (n:Node {id: node.id})
            SET n.type = node.type, n.label = node.label, n.color = node.color
            """,
            {"nodes": response_data["nodes"]},
        )
        # Import relationships
        relationships = self.driver.query(
            """
            UNWIND $rels AS rel
            MATCH (s:Node {id: rel.from})
            MATCH (t:Node {id: rel.to})
            MERGE (s)-[r:RELATIONSHIP {type:rel.relationship}]->(t)
            SET r.direction = rel.direction,
                r.color = rel.color,
                r.timestamp = timestamp()
            """,
            {"rels": response_data["edges"]},
        )
        return (nodes.result_set, relationships.result_set)

    @staticmethod
    def _process_graph_data(record) -> dict[str, Any]:
        """
        This function processes a record from the FalkorDB query result
        and formats it as a dictionary with the node details and the relationship.

        :param record: A record from the FalkorDB query result
        :return: A dictionary representing the graph data
        """
        try:
            node_from = record[0].properties
            relationship = record[1].properties
            node_to = record[2].properties

            graph_data = {
                "from_node": node_from,
                "relationship": relationship,
                "to_node": node_to,
            }

            return graph_data
        except Exception as e:
            return {"error": str(e)}

class Neo4j(Driver):
    def __init__(self):
        # If Neo4j credentials are set, then Neo4j is used to store information
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        url = os.environ.get("NEO4J_URI")
        if url is None:
            url = os.environ.get("NEO4J_URL")
            if url is not None:
                logging.warning("Obsolete: Please define NEO4J_URI instead")

        if username and password and url:
            self.driver = GraphDatabase.driver(url, auth=(username, password))
            # Check if connection is successful
            with self.driver.session() as session:
                try:
                    session.run("RETURN 1")
                    logging.info("Neo4j database connected successfully!")
                except ValueError as ve:
                    logging.error("Neo4j database: {}".format(ve))
                    raise
        else:
            raise ValueError("Configuration for Neo4j is missing")

    def get_graph_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes, _, _ = self.driver.execute_query(
            """
        MATCH (n)
        WITH collect(
            {data: {id: n.id, label: n.label, color: n.color}}) AS node
        RETURN node
        """
        )
        nodes = [el["node"] for el in nodes][0]

        edges, _, _ = self.driver.execute_query(
            """
        MATCH (s)-[r]->(t)
        WITH collect(
            {data: {source: s.id, target: t.id, label:r.type, color: r.color}}
        ) AS rel
        RETURN rel
        """
        )
        edges = [el["rel"] for el in edges][0]

        return (nodes, edges)

    def get_graph_history(self, skip, per_page) -> dict[str, Any]:
        # Getting the total number of graphs
        total_graphs, _, _ = self.driver.execute_query(
            """
        MATCH (n)-[r]->(m)
        RETURN count(n) as total_count
        """
        )
        total_count = total_graphs[0]["total_count"]

        # Fetching 10 most recent graphs
        result, _, _ = self.driver.execute_query(
            """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        ORDER BY r.timestamp DESC
        SKIP {skip}
        LIMIT {per_page}
        """.format(
            skip=skip, per_page=per_page
        )
        )

        # Process the 'result' to format it as a list of graphs
        graph_history = [Neo4j._process_graph_data(record) for record in result]
        remaining = max(0, total_count - skip - per_page)

        return {"graph_history": graph_history, "remaining": remaining, "graph": True}

    def get_response_data(
            self, response_data
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Import nodes
        nodes = self.driver.execute_query(
            """
            UNWIND $nodes AS node
            MERGE (n:Node {id: node.id})
            SET n.type = node.type, n.label = node.label, n.color = node.color
            """,
            {"nodes": response_data["nodes"]},
        )
        # Import relationships
        relationships = self.driver.execute_query(
            """
            UNWIND $rels AS rel
            MATCH (s:Node {id: rel.from})
            MATCH (t:Node {id: rel.to})
            MERGE (s)-[r:RELATIONSHIP {type:rel.relationship}]->(t)
            SET r.direction = rel.direction,
                r.color = rel.color,
                r.timestamp = timestamp();
            """,
            {"rels": response_data["edges"]},
        )
        return (nodes, relationships)

    @staticmethod
    def _process_graph_data(record):
        """
        This function processes a record from the Neo4j query result
        and formats it as a dictionary with the node details and the relationship.

        :param record: A record from the Neo4j query result
        :return: A dictionary representing the graph data
        """
        try:
            node_from = record["n"].items()
            node_to = record["m"].items()
            relationship = record["r"].items()

            graph_data = {
                "from_node": {key: value for key, value in node_from},
                "to_node": {key: value for key, value in node_to},
                "relationship": {key: value for key, value in relationship},
            }

            return graph_data
        except Exception as e:
            return {"error": str(e)}

def correct_json(json_str):
    """
    Corrects the JSON response from OpenAI to be valid JSON by removing trailing commas
    """
    while ",\s*}" in json_str or ",\s*]" in json_str:  # noqa: W605
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(
            "SanitizationError: %s for JSON: %s", str(e), json_str, exc_info=True
        )
        return None

