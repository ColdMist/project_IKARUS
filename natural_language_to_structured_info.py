import numpy as np
from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from utils.helper_functions import *
import pandas as pd


setup_openAI()
nlp = load_nlp()

text = """
    Graphs are an essential tool in material science for visualizing and analyzing various aspects of materials and their properties. By plotting data points and establishing relationships between variables, graphs enable researchers to gain insights and make informed decisions.
    One common type of graph used in material science is the stress-strain curve. This graph depicts the relationship between applied stress and resulting strain in a material during mechanical testing. By subjecting a material to controlled deformation, researchers can determine its mechanical properties, such as its yield strength, ultimate tensile strength, and modulus of elasticity. Plotting stress on the y-axis and strain on the x-axis, the stress-strain curve provides valuable information about a material's behavior under different loading conditions.
    Another useful graph in material science is the phase diagram. Phase diagrams illustrate the different phases or states of a material as a function of temperature and pressure. These diagrams help identify the stable phases and phase transitions that occur within a material. By plotting temperature on the x-axis and pressure on the y-axis, phase diagrams provide valuable insights into the solid, liquid, and gas phases of a material, as well as the conditions under which phase transformations occur.
    Graphs are also utilized in materials characterization techniques. For example, X-ray diffraction (XRD) data can be plotted as a graph showing the diffraction peaks and their corresponding angles. This graph allows researchers to identify the crystal structure and phase composition of a material. Similarly, scanning electron microscopy (SEM) and transmission electron microscopy (TEM) generate images that can be represented as graphs, revealing the microstructure and morphology of materials at various length scales.
    Furthermore, graphs are instrumental in studying material properties as a function of composition. For instance, phase diagrams known as binary or ternary phase diagrams are used to understand the relationship between different components in an alloy system. By plotting the composition of elements or alloys on the x-axis and the corresponding properties, such as hardness or melting point, on the y-axis, researchers can identify optimal compositions for desired material characteristics.
    In summary, graphs are invaluable in material science as they provide a visual representation of data, facilitating the interpretation and analysis of material properties. From stress-strain curves and phase diagrams to characterization techniques and composition-property relationships, graphs play a vital role in advancing our understanding of materials and driving innovations in the field.
    """

index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(text)

graph = index_creator.from_text(text)

triples = graph.get_triples()

print(f"detected triples are {triples}")
store_triples(triples, "data/triples.txt")
triples = pd.read_table("data/triples.txt", header=None)

# collect the unique entities
unique_entities = np.unique(
    (list(list(triples[0].unique()) + list(triples[2].unique())))
)
# print(f"unique entities are {unique_entities}")
connection_information = obtain_connection_information(nlp, unique_entities)
#store_to_pickle("data/connection_information.pkl", connection_information)

# TODO Need to store this into a file after proper serialization
