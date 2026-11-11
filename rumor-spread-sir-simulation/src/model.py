import mesa
import networkx as nx


# Each agent represents one person in the network
class RumorAgent(mesa.Agent):

    def __init__(self, model, node_id):
        super().__init__(model)
        self.node_id = node_id
        self.state = "S"

    # Called once per simulation step
    def step(self):
        if self.state == "I":
            for neighbor_id in self.model.G.neighbors(self.node_id):
                neighbor = self.model.node_to_agent[neighbor_id]
                if neighbor.state == "S":
                    if self.model.random.random() < self.model.beta:
                        neighbor.state = "I"
            if self.model.random.random() < self.model.gamma:
                self.state = "R"


# Handles network creation, agent setup and data collection
class RumorModel(mesa.Model):

    def __init__(self, n_agents=100, beta=0.05, gamma=0.3,
                 network_type="BA", seed=None):
        super().__init__(rng=seed)
        self.n_agents = n_agents
        self.beta = beta
        self.gamma = gamma

        if network_type == "BA":
            G = nx.barabasi_albert_graph(n=n_agents, m=2, seed=seed)
        else:
            G = nx.watts_strogatz_graph(n=n_agents, k=4, p=0.1, seed=seed)
        self.G = G

        self.node_to_agent = {}
        for node in G.nodes():
            agent = RumorAgent(self, node)
            self.node_to_agent[node] = agent

        seed_node = self.random.choice(list(G.nodes()))
        self.node_to_agent[seed_node].state = "I"

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "S": lambda m: sum(1 for a in m.agents if a.state == "S"),
                "I": lambda m: sum(1 for a in m.agents if a.state == "I"),
                "R": lambda m: sum(1 for a in m.agents if a.state == "R"),
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    # Run the simulation and return results 
    def run(self, steps=60):
        for _ in range(steps):
            self.step()
            if sum(1 for a in self.agents if a.state == "I") == 0:
                break
        return self.datacollector.get_model_vars_dataframe()
