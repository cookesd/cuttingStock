# cuttingStock

- Solves cutting stock problem using column generation
- Solves the column generation by solving a unbounded knapsack problem using dynamic programming (recursion)
- Currently solves for continuous valued number of cuts and uses rudimentary rounding up to ensure integrality
- The solution shows the integer and non-integer solution

- `classes` directory has two classes:
	- `CuttingStock` class to make the cutting stock problem with a method to solve it (method is in a separate script).
	- `CuttingStockSolution` class that is the output of the `solve` method of the `CuttingStock` class.

- In the top directory, there are two scripts I should try integrating with the above classes:
	- `init_gui.py` has a GUI layout that receives user input and displays the output, but references a different cutting stock class.
	- `pyo_model.py` is the beginning of a pyomo linear programming model representation of the problem. I way to solve differently than the revised simplex with knapsack formulation used in the `CuttingStock` class' `solve` method.