import pyreason as pr
import networkx as nx

pr.settings.verbose = True     # Print info to screen
pr.settings.atom_trace = True  # This allows us to view all the atoms that have made a certain rule fire

def add_rule(g):
    pr.load_graph(g)
    print("Done loading graph! :)")