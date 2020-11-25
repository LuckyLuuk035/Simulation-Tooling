## https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html

# Mesa imports:
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid # Je hebt SingleGrid en MultiGrid
                                 # Bij SingleGrid is er een max van 1 agent per cel
                                 # En bij MultiGrid kunnen er meerdere agents op een cel.
from mesa.datacollection import DataCollector # De methode om data bij elke stap op te slaan.
                                              # Hierdoor hoef je zelf geen loop te schrijven.
from mesa.batchrunner import BatchRunner


# Andere imports:
import matplotlib.pyplot as plt
import numpy as np

# Een functie die de ongelijkheid in wealth aangeeft
def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)) # De formule voor Gini Index/Ratio 
    return (1 + (1/N) - 2*B)



# Maak een class aan voor de Agent
class MoneyAgent(Agent):
    """ Een Agent met bepaald aantal begin wealth. """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True, # More is alle 8 vakjes er omheen en Von Neuman
                        # is alleen de vier boven onder links en rechts.
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
    
    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()



# Maak een class aan voor het Model       
class MoneyModel(Model):
    """ Een Model met bepaald aantal Agents. """
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True # Dit is voor Batch running

        # Create Agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"})
    
    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()


fixed_params = {"width": 10,
               "height": 10}
variable_params = {"N": range(10, 500, 10)}

batch_run = BatchRunner(MoneyModel,
                        variable_params,
                        fixed_params,
                        iterations=5,
                        max_steps=100,
                        model_reporters={"Gini": compute_gini})
batch_run.run_all()

plt.show() # Aangezien ik in de IDLE werk moet ik deze line neerzetten.
           # Dit zorgt er voor de weergave.
