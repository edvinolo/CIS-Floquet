import sys

sys.path.append('./src/')
from simulation import Simulation

def main():
    input = sys.argv[1]
    simulation = Simulation(input)
    simulation.run_simulation()

    return
   
#Run main function
main()