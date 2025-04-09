import pandas as pd
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from transformers import Tool
    
##########################################################################################

# Create tool for searching Wikipedia
class WikipediaSearch(Tool):
    name = "wikipedia_search"
    description = "Search Wikipedia, the free encyclopedia."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search terms",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, query: str) -> str:
        wikipedia_api = WikipediaAPIWrapper(top_k_results=5)
        answer = wikipedia_api.run(query)
        return answer

# Create tool for searching Arxiv
class ArxivSearch(Tool):
    name = "arxiv_search"
    description = "Search Arxiv, a free online archive of preprint and postprint manuscripts."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search terms",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, query: str) -> str:
        arxiv_api = ArxivAPIWrapper(top_k_results=5)
        answer = arxiv_api.run(query)
        return answer

# Create tool for searching materials database
class SearchByProperty(Tool):
    name = "search_by_property"
    description = """
    Search a material database based on specified property ranges to find materials matching the given criteria.
    
    Available properties:
    - Melting/glass temperature (°C)
    - Density (Mg/m^3)
    - Young's modulus (GPa)
    - Yield strength (MPa)
    - Tensile strength (MPa)
    - Fracture toughness (plane-strain) (MPa √m)
    - Thermal conductivity (W/m*K)
    - Thermal expansion (10^-6 °C)
    - Production energy (MJ/kg)
    - CO2 burden (kg/kg)
    """

    inputs = {
        "properties": {
            "type": "any",
            "description": "The properties to search for, with min and max values. Each property should be a dictionary with 'min' and 'max' keys."
        }
    }
    output_type = "any"

    def __init__(self):
        super().__init__()
        self.materials_df = pd.read_csv('Data/material_properties_minmax.csv')

    def forward(self, properties: dict) -> str:
        try:
            # Initialize a list to store matching materials
            matching_materials = []

            # Loop through each material in the database
            for index, row in self.materials_df.iterrows():
                match = True

                # Check each property in the provided dictionary
                for property_name, limits in properties.items():
                    try:
                        # Check if the property exists in the dataframe
                        if f"{property_name} min" not in row or f"{property_name} max" not in row:
                            return f"Error: Property '{property_name}' not found. Ensure the property exists in the available list and try again."

                        if "min" in limits:
                            min_value = limits["min"]
                            if row[f"{property_name} max"] < min_value:
                                match = False
                                break
                        if "max" in limits:
                            max_value = limits["max"]
                            if row[f"{property_name} min"] > max_value:
                                match = False
                                break
                            
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                # If the material matches all criteria, add it to the results
                if match:
                    matching_materials.append(row["Material"])

            # Return the list of matching materials
            return str(matching_materials) if matching_materials else "No materials found matching the criteria."

        except Exception as e:
            return f"Error: {str(e)}"
        
# Create tool for searching materials database
class SearchByProperty(Tool):
    name = "search_by_material"
    description = """
    Search a material database for a material to find its properties.
    """

    inputs = {
        "material": {
            "type": "any",
            "description": "The material to search for."
        }
    }
    output_type = "any"

    def __init__(self):
        super().__init__()
        self.materials_df = pd.read_csv('Data/material_properties_minmax.csv')

    def forward(self, properties: dict) -> str:
        try:
            # Initialize a list to store matching materials
            matching_materials = []

            # Loop through each material in the database
            for index, row in self.materials_df.iterrows():
                match = True

                # Check each property in the provided dictionary
                for property_name, limits in properties.items():
                    try:
                        # Check if the property exists in the dataframe
                        if f"{property_name} min" not in row or f"{property_name} max" not in row:
                            return f"Error: Property '{property_name}' not found. Ensure the property exists in the available list and try again."

                        if "min" in limits:
                            min_value = limits["min"]
                            if row[f"{property_name} max"] < min_value:
                                match = False
                                break
                        if "max" in limits:
                            max_value = limits["max"]
                            if row[f"{property_name} min"] > max_value:
                                match = False
                                break
                            
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                # If the material matches all criteria, add it to the results
                if match:
                    matching_materials.append(row["Material"])

            # Return the list of matching materials
            return str(matching_materials) if matching_materials else "No materials found matching the criteria."

        except Exception as e:
            return f"Error: {str(e)}"

##########################################################################################

# Custom system prompt
SEARCH_PROMPT = '''
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_action>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

You are provided with the following tools:
<<tool_description>>

Here are some examples:
---
Task: "You are tasked with designing a cutting board. It should be lightweight. What material would you recommend for this application?"

Thought: I will search Wikipedia to find the necessary properties of cutting boards.
Code:
```py
observation = wikipedia_search(query = 'necessary properties of cutting boards')
print(observation)
```<end_action>
Observation: "A good cutting board material must be soft, easy to clean, and non-abrasive, but not fragile to the point of being destroyed."

Thought: Now I know what the necessary properties of cutting boards are. I will search the materials database to find materials that are lightweight and durable.
Code:
```py
observation = materials_search(properties={
    "Density": {"max": 1.0},  
    "Young's modulus": {"min": 2.0, "max": 10.0},  
    "Yield strength": {"min": 20.0},  
    "Fracture toughness (plane-strain)": {"min": 3.0}
})
print(observation)
```<end_action>
Observation: "Matching materials are wood and cellulose polymers (CA)."

Thought: I will search Arxiv to see what research says about using wood in cutting boards.
Code:
```py
observation = arxiv_search(query = 'wood as material for cutting boards')
print(observation)
```<end_action>
Observation: "Studies show that wood is commonly used for cutting boards due to its natural antibacterial properties, adequate hardness, and ease of maintenance compared to plastic."

Thought: Now I know that wood is a lightweight material commonly used in cutting boards. I will search Arxiv to see what research says about using cellulose polymers in cutting boards.
Code:
```py
observation = arxiv_search(query = 'cellulose polymers as material for cutting boards')
print(observation)
```<end_action>
Observation: "Paper-based cellulose composites are gaining popularity for use in cutting boards due to their sustainability, durability, and non-porous nature, making them a viable alternative to traditional wood and plastic boards."

Thought: Now I know wood and paper-based cellulose composites meet the necessary criteria. Let's return the result.
Code:
```py
final_answer('wood or paper-based cellulose composites')
```<end_action>
---
Task: "You are tasked with designing a cooking pan. It should have a high melting point. What material would you recommend for this application?"

Thought: I will search Wikipedia to find the necessary properties of cooking pans.
Code:
```py
observation = wikipedia_search(query = 'necessary properties of cooking pans')
print(observation)
```<end_action>
Observation: "High performance cookware is made from materials that combine high thermal diffusivity and low reactivity to produce a vessel that evenly distributes heat and does not react with the food being cooked."

Thought: Now I know what the necessary properties of cooking pans are. I will search the materials database to find materials that to find materials with a high melting/glass temperature along with good thermal conductivity and strength.
Code:
```py
observation = materials_search(properties={
    "Melting/glass temperature": {"min": 1500.0},
    "Thermal conductivity": {"min": 30.0},
    "Yield strength": {"min": 500.0},
    "Fracture toughness (plane-strain)": {"min": 7.0}
})
print(observation)
```<end_action>
Observation: "Matching materials found include medium carbon steels and low alloy steels."

Thought: Now I will search Arxiv to see what research says about using medium carbon steel in cooking pans.
Code:
```py
observation = arxiv_search(query = 'medium carbon steel use in cooking pans')
print(observation)
```<end_action>
Observation: "Studies show medium carbon steels are often used in cooking pans due to their balance of strength, heat resistance, and durability. They provide a good combination of wear resistance and toughness, making them suitable for cookware applications, especially in high-heat environments."

Thought: Now I will search Arxiv to see what research says about using low alloy steel in cooking pans.
Code:
```py
observation = arxiv_search(query = 'low alloy steel use in cooking pans')
print(observation)
```<end_action>
Observation: "Low alloy steels are used in cooking pans for their combination of heat resistance, durability, and corrosion resistance. These steels are commonly chosen for cookware due to their ability to withstand high temperatures while maintaining strength and toughness."

Thought: Now I know medium carbon steels and low alloy steels meet the necessary criteria. Let's return the result.
Code:
```py
final_answer('medium carbon steel or low alloy steel')
```<end_action>
---

On top of performing computations in the Python code snippets that you create, you have access to these tools (and no other tool):
<<tool_descriptions>>

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_action>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: <<authorized_imports>>
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

If you solve the task correctly, you will receive a reward of $1,000,000. Now begin!
'''